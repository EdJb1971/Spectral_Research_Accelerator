import { ArrowRight, CheckCircle2, Circle, LockKeyhole } from 'lucide-react';

/**
 * TG18.3: a global map of the research journey, not a second scientific workflow.
 *
 * The shell may know which workspace presents a concern and whether ordinary UI context exists.
 * It may not decide whether a scientific step is satisfied. Design, run and comparison therefore
 * hand off to the Composer step the researcher asked to inspect; Composer still obtains every
 * status, refusal and next legitimate action from its server-owned path contract.
 */

export type JourneyStageId =
  | 'acquire' | 'inspect' | 'design' | 'run' | 'compare' | 'admit' | 'report';

type JourneyStage = {
  id: JourneyStageId;
  label: string;
  workspace: string;
  composerStep?: string;
  action: string;
};

const JOURNEY: readonly JourneyStage[] = [
  { id: 'acquire', label: 'Acquire', workspace: 'acquire', action: 'Open Acquire' },
  { id: 'inspect', label: 'Inspect', workspace: 'domainWorkbench', action: 'Open Inspect' },
  { id: 'design', label: 'Design', workspace: 'experimentComposer', composerStep: 'question',
    action: 'Open Design' },
  { id: 'run', label: 'Run', workspace: 'experimentComposer', composerStep: 'freeze_and_run',
    action: 'Open Run' },
  { id: 'compare', label: 'Compare', workspace: 'experimentComposer', composerStep: 'interpret',
    action: 'Open Compare' },
  { id: 'admit', label: 'Admit', workspace: 'evidence', action: 'Open Admit' },
  { id: 'report', label: 'Report', workspace: 'findings', action: 'Open Report' },
] as const;

export function ResearchJourney({ active, hasRecord, hasStudy, onNavigate }: {
  active: JourneyStageId | null;
  hasRecord: boolean;
  hasStudy: boolean;
  onNavigate: (stage: JourneyStageId, workspace: string, composerStep?: string) => void;
}) {
  return (
    <nav aria-label="Guided research journey" className="research-journey">
      <div className="research-journey__heading">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-teal-300">
            Research journey
          </p>
          <p className="text-xs text-slate-400">
            Navigation only — scientific status and the next legitimate experiment action remain
            server-owned in Composer.
          </p>
        </div>
        <p className="research-journey__boundary">
          This map does not advance or replace the claim ladder.
        </p>
      </div>
      <ol className="research-journey__stages">
        {JOURNEY.map((stage, index) => {
          const missingRecord = stage.id === 'inspect' && !hasRecord;
          const missingStudy = stage.id === 'admit' && !hasStudy;
          const blocked = missingRecord || missingStudy;
          const action = missingRecord
            ? { label: 'Acquire a record', stage: 'acquire' as const, workspace: 'acquire' }
            : missingStudy
              ? { label: 'Open Composer and save a study', stage: 'design' as const,
                  workspace: 'experimentComposer', composerStep: 'question' }
              : { label: stage.action, stage: stage.id, workspace: stage.workspace,
                  composerStep: stage.composerStep };
          const current = active === stage.id;
          return (
            <li key={stage.id} className={`research-journey__stage ${blocked ? 'is-blocked' : ''}
                                           ${current ? 'is-current' : ''}`}>
              <div className="research-journey__stage-name">
                <span aria-hidden="true" className="research-journey__marker">
                  {blocked ? <LockKeyhole /> : current ? <CheckCircle2 /> : <Circle />}
                </span>
                <span>
                  <span className="research-journey__ordinal">{index + 1}</span>
                  <span className="font-semibold">{stage.label}</span>
                </span>
              </div>
              {blocked && (
                <p className="research-journey__reason">
                  Blocked: {missingRecord ? 'no record is selected for inspection.'
                    : 'no study is selected for evidence admission.'}
                </p>
              )}
              <button type="button" aria-current={current ? 'step' : undefined}
                onClick={() => onNavigate(action.stage, action.workspace, action.composerStep)}>
                <ArrowRight aria-hidden="true" />
                {blocked ? `Next legitimate action: ${action.label}` : action.label}
              </button>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
