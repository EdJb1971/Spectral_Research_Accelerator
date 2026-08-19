import React, { useState } from 'react';

interface LineageNode {
  id: string;
  name: string;
  type: string;
  value: Record<string, any>;
  created_at: string;
}

interface LineageEdge {
  id: string;
  source_id: string;
  target_id: string;
  relation: string;
}

interface LineageGraphProps {
  nodes: LineageNode[];
  edges: LineageEdge[];
}

export const LineageGraph: React.FC<LineageGraphProps> = ({ nodes, edges }) => {
  const [selectedNode, setSelectedNode] = useState<LineageNode | null>(null);

  if (!nodes || nodes.length === 0) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-6 flex items-center justify-center h-64 text-slate-500 w-full">
        No lineage data available for this experiment.
      </div>
    );
  }

  const columns: LineageNode[][] = [[], [], [], []];
  nodes.forEach(node => {
    if (node.type === 'code_revision' || node.type === 'dataset') {
      columns[0].push(node);
    } else if (node.type === 'field') {
      columns[1].push(node);
    } else if (node.type === 'coefficients') {
      columns[2].push(node);
    } else {
      columns[3].push(node);
    }
  });

  const width = 800;
  const height = 400;
  const colWidth = width / 4;

  const nodePositions: Record<string, { x: number; y: number }> = {};
  columns.forEach((colNodes, colIdx) => {
    const x = colIdx * colWidth + colWidth / 2;
    const count = colNodes.length;
    colNodes.forEach((node, nodeIdx) => {
      const y = ((nodeIdx + 1) * height) / (count + 1);
      nodePositions[node.id] = { x, y };
    });
  });

  const getNodeColor = (type: string) => {
    switch (type) {
      case 'code_revision': return '#ec4899';
      case 'dataset': return '#3b82f6';
      case 'field': return '#10b981';
      case 'coefficients': return '#f59e0b';
      case 'metrics': return '#8b5cf6';
      default: return '#64748b';
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 bg-slate-950 p-4 rounded-xl border border-slate-800 w-full">
      <div className="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-lg p-4 relative overflow-x-auto">
        <h3 className="text-sm font-semibold text-slate-300 mb-4">Provenance Lineage Graph</h3>
        <svg viewBox={`0 0 ${width} ${height}`} className="min-w-[600px] w-full h-auto">
          {edges.map((edge, idx) => {
            const start = nodePositions[edge.source_id];
            const end = nodePositions[edge.target_id];
            if (!start || !end) return null;

            const dx = end.x - start.x;
            const controlX1 = start.x + dx * 0.4;
            const controlX2 = start.x + dx * 0.6;

            return (
              <g key={`edge-${idx}`}>
                <path
                  d={`M ${start.x} ${start.y} C ${controlX1} ${start.y}, ${controlX2} ${end.y}, ${end.x} ${end.y}`}
                  fill="none"
                  stroke="#334155"
                  strokeWidth="2"
                  markerEnd="url(#arrow)"
                />
                <text
                  x={(start.x + end.x) / 2}
                  y={(start.y + end.y) / 2 - 4}
                  fill="#475569"
                  fontSize="8"
                  textAnchor="middle"
                >
                  {edge.relation}
                </text>
              </g>
            );
          })}

          <defs>
            <marker
              id="arrow"
              viewBox="0 0 10 10"
              refX="16"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#334155" />
            </marker>
          </defs>

          {nodes.map(node => {
            const pos = nodePositions[node.id];
            if (!pos) return null;

            const color = getNodeColor(node.type);
            const isSelected = selectedNode?.id === node.id;

            return (
              <g
                key={node.id}
                transform={`translate(${pos.x}, ${pos.y})`}
                className="cursor-pointer"
                onClick={() => setSelectedNode(node)}
              >
                <circle
                  r={isSelected ? "18" : "14"}
                  fill="#0f172a"
                  stroke={color}
                  strokeWidth={isSelected ? "4" : "2"}
                  className="transition-all duration-200"
                />
                <text
                  fill="#f1f5f9"
                  fontSize="9"
                  fontWeight="bold"
                  textAnchor="middle"
                  y="3"
                >
                  {node.type.substring(0, 2).toUpperCase()}
                </text>
                <text
                  fill="#94a3b8"
                  fontSize="8"
                  textAnchor="middle"
                  y="28"
                >
                  {node.name.length > 12 ? `${node.name.substring(0, 10)}...` : node.name}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 flex flex-col justify-between">
        <div>
          <h3 className="text-sm font-semibold text-slate-300 mb-4">Node Inspector</h3>
          {selectedNode ? (
            <div className="space-y-4">
              <div>
                <span className="text-xs text-slate-500 block">Node ID</span>
                <span className="text-sm font-mono text-slate-300">{selectedNode.id}</span>
              </div>
              <div>
                <span className="text-xs text-slate-500 block">Name</span>
                <span className="text-sm font-semibold text-slate-200">{selectedNode.name}</span>
              </div>
              <div>
                <span className="text-xs text-slate-500 block">Type</span>
                <span
                  className="text-xs px-2 py-0.5 rounded font-mono inline-block mt-1"
                  style={{
                    backgroundColor: `${getNodeColor(selectedNode.type)}20`,
                    color: getNodeColor(selectedNode.type),
                    border: `1px solid ${getNodeColor(selectedNode.type)}40`
                  }}
                >
                  {selectedNode.type}
                </span>
              </div>
              <div>
                <span className="text-xs text-slate-500 block">Created At</span>
                <span className="text-sm text-slate-300">{new Date(selectedNode.created_at).toLocaleString()}</span>
              </div>
              <div>
                <span className="text-xs text-slate-500 block mb-1">Metadata / Values</span>
                <pre className="text-xs bg-slate-950 p-2 rounded border border-slate-800 max-h-48 overflow-y-auto text-teal-400 font-mono">
                  {JSON.stringify(selectedNode.value, null, 2)}
                </pre> 
              </div>
            </div>
          ) : ( 
            <div className="text-sm text-slate-500 text-center py-12">
              Select a node in the graph to inspect its provenance details.
            </div>
          )}
        </div>
        {selectedNode && (
          <div className="mt-4 pt-4 border-t border-slate-800 flex justify-end">
            <button
              onClick={() => setSelectedNode(null)}
              className="text-xs text-slate-400 hover:text-slate-200 px-3 py-1.5 rounded border border-slate-800 hover:bg-slate-800 transition-colors"
            >
              Clear Selection
            </button>
          </div>
        )}
      </div>
    </div>
  );
};