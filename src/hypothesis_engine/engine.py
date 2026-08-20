import uuid
import datetime
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from src.statistics.significance import correlation_test, screen
from src.database.models import Experiment, ExperimentRun, Hypothesis

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

class PatternDiscoveryEngine:
    @classmethod
    def discover_patterns(
        cls,
        db: Session,
        experiment_ids: Optional[List[str]] = None,
        target_metrics: Optional[List[str]] = None,
        confidence_threshold: float = 0.3,
        alpha: float = 0.05,
        correction: str = "benjamini_yekutieli",
        require_significance: bool = True
    ) -> List[Hypothesis]:
        """Scan for patterns, then apply multiplicity control before reporting any.

        **Defect D8.** This method previously reported every parameter-metric pair whose
        |r| exceeded `confidence_threshold`, with no p-value at all. On the 9-run smoke sweep
        that produced 9 "discoveries", including *"strong positive correlation (r = 0.96)
        between grid size and floating-point reconstruction error"*. An engine whose purpose
        is to test many hypotheses will always find some at a fixed effect-size threshold -
        on any data, including noise.

        Now every candidate is tested, the **whole family** is corrected together, and only
        survivors are returned. `require_significance=False` returns the failures too, marked
        `significant=False`, which is useful for auditing a scan but must never be used to
        report a finding.

        The default correction is Benjamini-Yekutieli, which controls FDR under *arbitrary*
        dependence. Parameter-metric pairs drawn from the same runs are dependent in ways
        that are not guaranteed to be positive, and BY is the procedure that stays valid
        without assuming otherwise. It costs power; assuming PRDS because it is more
        convenient would cost validity.
        """
        # Query completed runs
        query = db.query(ExperimentRun).filter(ExperimentRun.status == "COMPLETED")
        if experiment_ids:
            query = query.filter(ExperimentRun.experiment_id.in_(experiment_ids))
        
        runs = query.all()
        if not runs:
            return []
        
        # Group runs by experiment_id
        runs_by_exp: Dict[str, List[ExperimentRun]] = {}
        for r in runs:
            runs_by_exp.setdefault(r.experiment_id, []).append(r)
            
        discovered_hypotheses: List[Hypothesis] = []
        
        for exp_id, exp_runs in runs_by_exp.items():
            if len(exp_runs) < 2:
                continue
                
            experiment = db.query(Experiment).filter(Experiment.id == exp_id).first()
            if not experiment:
                continue
                
            flat_results_list = []
            parameters_list = []
            for r in exp_runs:
                parameters_list.append(r.parameters)
                flat_results_list.append(flatten_dict(r.results or {}))
                
            param_keys = set()
            for p in parameters_list:
                param_keys.update(p.keys())
                
            metric_keys = set()
            for res in flat_results_list:
                metric_keys.update(res.keys())
                
            if target_metrics:
                filtered_metrics = []
                for m in metric_keys:
                    if any(tm in m for tm in target_metrics):
                        filtered_metrics.append(m)
                metric_keys = set(filtered_metrics)
                
            numerical_params = []
            categorical_params = []
            for pk in param_keys:
                is_num = True
                values = []
                for p in parameters_list:
                    val = p.get(pk)
                    if val is None:
                        is_num = False
                        break
                    try:
                        values.append(float(val))
                    except (ValueError, TypeError):
                        is_num = False
                        break
                if is_num and len(set(values)) > 1:
                    numerical_params.append(pk)
                elif len(set(str(p.get(pk)) for p in parameters_list)) > 1:
                    categorical_params.append(pk)
                    
            numerical_metrics = []
            for mk in metric_keys:
                is_num = True
                values = []
                for res in flat_results_list:
                    val = res.get(mk)
                    if val is None:
                        is_num = False
                        break
                    try:
                        values.append(float(val))
                    except (ValueError, TypeError):
                        is_num = False
                        break
                if is_num and len(set(values)) > 1:
                    numerical_metrics.append(mk)
                    
            # 1. Numerical parameter vs numerical metric: collect every test first, so the
            #    correction sees the true family size rather than a post-hoc selection.
            candidates = []
            for pk in numerical_params:
                for mk in numerical_metrics:
                    p_vals = []
                    m_vals = []
                    for p, res in zip(parameters_list, flat_results_list):
                        p_vals.append(float(p[pk]))
                        m_vals.append(float(res[mk]))
                        
                    p_arr = np.array(p_vals)
                    m_arr = np.array(m_vals)
                    
                    std_p = np.std(p_arr)
                    std_m = np.std(m_arr)
                    if std_p == 0 or std_m == 0:
                        continue
                        
                    r = np.corrcoef(p_arr, m_arr)[0, 1]
                    if np.isnan(r):
                        continue
                        
                    test = correlation_test(p_arr, m_arr,
                                            account_for_autocorrelation=False)
                    candidates.append({
                        "kind": "correlation",
                        "label": "corr:%s~%s" % (pk, mk),
                        "p_value": test["p_value"],
                        "r": r,
                        "pk": pk,
                        "mk": mk,
                        "p_vals": p_vals,
                        "m_vals": m_vals,
                        "n": test["n"],
                        "test": test,
                    })

            # --- multiplicity control over the WHOLE family, before anything is reported ---
            screening = screen(candidates, alpha=alpha, method=correction,
                               n_tests=len(candidates))
            by_label = {c["label"]: c for c in screening["results"]}

            for cand in screening["results"]:
                if cand["kind"] != "correlation":
                    continue
                r = cand["r"]
                pk, mk = cand["pk"], cand["mk"]
                p_vals, m_vals = cand["p_vals"], cand["m_vals"]
                if require_significance and not cand["significant"]:
                    continue
                if abs(r) >= confidence_threshold:
                        pattern_type = "correlation"
                        direction = "positive" if r > 0 else "negative"
                        strength = "strong" if abs(r) >= 0.5 else "moderate"
                        
                        desc = (
                            f"In experiment '{experiment.name}', a {strength} {direction} correlation "
                            f"(r = {r:.2f}, p = {cand['p_value']:.3g}, q = {cand['q_value']:.3g} "
                            f"after {correction} over {screening['n_tests']} tests) "
                            f"between parameter '{pk}' and metric '{mk}'. "
                            f"n = {cand['n']} runs."
                        )
                        
                        proposed_config = cls._propose_numerical_followup(
                            experiment, pk, p_vals, mk, r
                        )
                        
                        hypothesis = Hypothesis(
                            id=str(uuid.uuid4()),
                            experiment_ids=[exp_id],
                            pattern_type=pattern_type,
                            description=desc,
                            confidence=float(abs(r)),
                            metrics_analyzed=[mk],
                            parameters_analyzed=[pk],
                            proposed_experiment_config=proposed_config,
                            p_value=float(cand["p_value"]),
                            q_value=float(cand["q_value"]),
                            n_tests=int(screening["n_tests"]),
                            statistics={
                                "test": "pearson_correlation",
                                "r": float(r),
                                "n": int(cand["n"]),
                                "p_value": float(cand["p_value"]),
                                "q_value": float(cand["q_value"]),
                                "significant": bool(cand["significant"]),
                                "correction": screening["correction"],
                                "alpha": alpha,
                                "assumptions": cand["test"]["assumptions"],
                                "caveat": (
                                    "Sweep parameters are chosen, not sampled, so this is an "
                                    "association across a designed grid rather than an "
                                    "estimate of a population correlation. It is not causal "
                                    "(rule R7)."),
                            }
                        )
                        db.add(hypothesis)
                        discovered_hypotheses.append(hypothesis)
                        
            # 2. Categorical Parameter vs. Numerical Metric
            # --- categorical: an ANOVA per pair, screened in its own family ---------
            categorical_tests = []
            for pk in categorical_params:
                for mk in numerical_metrics:
                    groups = {}
                    for prm, res in zip(parameters_list, flat_results_list):
                        groups.setdefault(str(prm[pk]), []).append(float(res[mk]))
                    usable = [v for v in groups.values() if len(v) >= 2]
                    if len(usable) < 2:
                        continue
                    try:
                        from scipy import stats as _st
                        f_stat, p_val = _st.f_oneway(*usable)
                        if not np.isfinite(p_val):
                            continue
                    except Exception:
                        continue
                    categorical_tests.append({
                        "kind": "categorical", "label": "anova:%s~%s" % (pk, mk),
                        "p_value": float(p_val), "f": float(f_stat), "pk": pk, "mk": mk})

            cat_screening = screen(categorical_tests, alpha=alpha, method=correction,
                                   n_tests=len(categorical_tests)) if categorical_tests else None
            cat_lookup = ({c["label"]: c for c in cat_screening["results"]}
                          if cat_screening else {})

            for pk in categorical_params:
                for mk in numerical_metrics:
                    cat_groups = {}
                    for p, res in zip(parameters_list, flat_results_list):
                        cat = str(p.get(pk))
                        val = float(res[mk])
                        cat_groups.setdefault(cat, []).append(val)
                        
                    if len(cat_groups) < 2:
                        continue
                        
                    cat_means = {cat: np.mean(vals) for cat, vals in cat_groups.items()}
                    
                    is_error = any(em in mk.lower() for em in ["error", "mse", "rmse", "mae", "leakage", "shift", "bias"])
                    
                    sorted_cats = sorted(cat_means.items(), key=lambda x: x[1], reverse=not is_error)
                    best_cat, best_val = sorted_cats[0]
                    worst_cat, worst_val = sorted_cats[-1]
                    
                    overall_mean = np.mean([val for vals in cat_groups.values() for val in vals])
                    if overall_mean == 0:
                        overall_mean = 1.0
                        
                    rel_diff = abs(best_val - worst_val) / abs(overall_mean)
                    
                    screened = cat_lookup.get("anova:%s~%s" % (pk, mk))
                    cat_p = float(screened["p_value"]) if screened else None
                    cat_q = float(screened["q_value"]) if screened else None
                    cat_significant = bool(screened["significant"]) if screened else False
                    if require_significance and not cat_significant:
                        continue

                    if rel_diff >= 0.1:
                        pattern_type = "categorical_opt"
                        desc = (
                            f"In experiment '{experiment.name}', categorical parameter '{pk}' is associated with '{mk}'. "
                            f"Category '{best_cat}' performed best with a mean of {best_val:.4f}, "
                            f"outperforming '{worst_cat}' (mean of {worst_val:.4f}) by {rel_diff*100:.1f}%. "
                            + (f"ANOVA p = {cat_p:.3g}, q = {cat_q:.3g} after {correction} "
                               f"over {len(categorical_tests)} tests."
                               if cat_p is not None else "")
                        )
                        
                        proposed_config = cls._propose_categorical_followup(
                            experiment, pk, best_cat
                        )
                        
                        hypothesis = Hypothesis(
                            id=str(uuid.uuid4()),
                            experiment_ids=[exp_id],
                            pattern_type=pattern_type,
                            description=desc,
                            confidence=float(min(1.0, rel_diff)),
                            metrics_analyzed=[mk],
                            parameters_analyzed=[pk],
                            proposed_experiment_config=proposed_config,
                            p_value=cat_p,
                            q_value=cat_q,
                            n_tests=int(screening["n_tests"]) + len(categorical_tests),
                            statistics={
                                "test": "one_way_anova",
                                "relative_difference": float(rel_diff),
                                "p_value": cat_p,
                                "q_value": cat_q,
                                "significant": cat_significant,
                                "correction": correction,
                                "alpha": alpha,
                                "caveat": (
                                    "A group-mean difference across a designed grid. Not "
                                    "causal (R7), and the groups are the levels the sweep "
                                    "happened to include."),
                            }
                        )
                        db.add(hypothesis)
                        discovered_hypotheses.append(hypothesis)
                        
        db.commit()
        return discovered_hypotheses

    @classmethod
    def _propose_numerical_followup(
        cls,
        experiment: Experiment,
        param_key: str,
        param_values: List[float],
        metric_key: str,
        r: float
    ) -> Dict[str, Any]:
        is_error = any(em in metric_key.lower() for em in ["error", "mse", "rmse", "mae", "leakage", "shift", "bias"])
        want_larger = (is_error and r < 0) or (not is_error and r > 0)
        
        min_val = min(param_values)
        max_val = max(param_values)
        span = max_val - min_val
        if span == 0:
            span = abs(min_val) if min_val != 0 else 1.0
            
        if want_larger:
            new_min = max_val
            new_max = max_val + span * 1.5
            new_values = [float(new_min), float(new_min + span * 0.75), float(new_max)]
        else:
            if min_val > 0:
                new_values = [float(min_val * 0.1), float(min_val * 0.3), float(min_val * 0.6)]
            else:
                new_min = min_val - span * 1.5
                new_max = min_val
                new_values = [float(new_min), float(new_min + span * 0.75), float(new_max)]
                
        orig_config = experiment.config
        new_matrix = {**orig_config.get("parameter_matrix", {})}
        new_matrix[param_key] = new_values
        
        direction_str = "larger" if want_larger else "smaller"
        
        followup_config = {
            "name": f"Follow-up: {experiment.name} (Optimizing {param_key})",
            "description": (
                f"Proposed automatically to explore the {direction_str} range of '{param_key}' "
                f"since it showed a {'negative' if r < 0 else 'positive'} correlation with '{metric_key}'."
            ),
            "parameter_matrix": new_matrix,
            "pipeline": orig_config.get("pipeline", []),
            "metadata": {
                "code_revision": orig_config.get("metadata", {}).get("code_revision", "unknown"),
                "dataset_version": orig_config.get("metadata", {}).get("dataset_version", "unknown"),
                "parent_experiment_id": experiment.id
            }
        }
        return followup_config

    @classmethod
    def _propose_categorical_followup(
        cls,
        experiment: Experiment,
        param_key: str,
        best_category: str
    ) -> Dict[str, Any]:
        orig_config = experiment.config
        new_matrix = {**orig_config.get("parameter_matrix", {})}
        new_matrix[param_key] = [best_category]
        
        followup_config = {
            "name": f"Follow-up: {experiment.name} (Fixed {param_key} to {best_category})",
            "description": (
                f"Proposed automatically to fix categorical parameter '{param_key}' to its best performing "
                f"category '{best_category}' and run other parameter combinations."
            ),
            "parameter_matrix": new_matrix,
            "pipeline": orig_config.get("pipeline", []),
            "metadata": {
                "code_revision": orig_config.get("metadata", {}).get("code_revision", "unknown"),
                "dataset_version": orig_config.get("metadata", {}).get("dataset_version", "unknown"),
                "parent_experiment_id": experiment.id
            }
        }
        return followup_config
