/**
 * AgentBase Knowledge Graph — D3.js Force-Directed Graph (v2)
 */
const AgentBaseGraph = (function() {
    let simulation = null;
    let svg = null;
    let g = null;
    let linkElements = null;
    let nodeElements = null;
    let labelElements = null;
    let linkLabelElements = null;
    let nodes = [];
    let edges = [];
    let width = 0;
    let height = 0;

    const TYPE_COLORS = {
        person: '#e15759',
        project: '#6366f1',
        concept: '#06b6d4',
        tool: '#f59e0b',
        event: '#22c55e',
        organization: '#8b5cf6',
        technology: '#ec4899',
        location: '#14b8a6',
        product: '#f97316',
    };

    function getColor(entityType) {
        return TYPE_COLORS[entityType] || '#94a3b8';
    }

    function getRadius(entityType) {
        // Vary node radius by type
        const base = {person: 14, project: 16, concept: 12, tool: 13, organization: 15, technology: 12};
        return base[entityType] || 11;
    }

    function init(selector, nodesData, edgesData) {
        const container = document.querySelector(selector);
        width = container.clientWidth;
        height = container.clientHeight || 600;

        const nodeMap = {};
        nodes = nodesData.map(n => ({...n}));
        nodes.forEach(n => { nodeMap[n.id] = n; });
        edges = edgesData.map(e => ({
            ...e,
            source: nodeMap[e.source] || e.source,
            target: nodeMap[e.target] || e.target,
        })).filter(e => typeof e.source === 'object' && typeof e.target === 'object');

        svg = d3.select(selector).append('svg')
            .attr('width', width).attr('height', height);

        g = svg.append('g');

        // Zoom
        svg.call(d3.zoom().scaleExtent([0.2, 6]).on('zoom', (event) => {
            g.attr('transform', event.transform);
        }));

        // Arrow marker
        svg.append('defs').append('marker')
            .attr('id', 'arrowhead')
            .attr('viewBox', '-0 -5 10 10')
            .attr('refX', 24).attr('refY', 0)
            .attr('orient', 'auto')
            .attr('markerWidth', 7).attr('markerHeight', 7)
            .append('path')
            .attr('d', 'M 0,-5 L 10,0 L 0,5')
            .attr('fill', '#94a3b8');

        // Links
        linkElements = g.append('g')
            .selectAll('line')
            .data(edges).enter().append('line')
            .attr('stroke', '#cbd5e1').attr('stroke-width', 1.5)
            .attr('marker-end', 'url(#arrowhead)');

        // Link labels
        linkLabelElements = g.append('g')
            .selectAll('text')
            .data(edges).enter().append('text')
            .attr('font-size', '10px').attr('fill', '#94a3b8')
            .attr('text-anchor', 'middle')
            .attr('dy', -4)
            .text(d => d.predicate);

        // Nodes
        nodeElements = g.append('g')
            .selectAll('circle')
            .data(nodes).enter().append('circle')
            .attr('r', d => getRadius(d.entity_type))
            .attr('fill', d => getColor(d.entity_type))
            .attr('stroke', '#fff').attr('stroke-width', 2.5)
            .style('cursor', 'pointer')
            .style('filter', 'drop-shadow(0 1px 2px rgba(0,0,0,0.15))')
            .call(d3.drag()
                .on('start', dragStarted)
                .on('drag', dragged)
                .on('end', dragEnded))
            .on('click', nodeClicked)
            .on('mouseover', function() { d3.select(this).attr('stroke-width', 4).style('filter', 'drop-shadow(0 2px 6px rgba(0,0,0,0.25))'); })
            .on('mouseout', function() { d3.select(this).attr('stroke-width', 2.5).style('filter', 'drop-shadow(0 1px 2px rgba(0,0,0,0.15))'); });

        // Node labels
        labelElements = g.append('g')
            .selectAll('text')
            .data(nodes).enter().append('text')
            .attr('font-size', '11px').attr('fill', '#1e293b')
            .attr('font-weight', '600')
            .attr('text-anchor', 'middle')
            .attr('dy', d => -getRadius(d.entity_type) - 6)
            .text(d => d.name);

        // Simulation
        simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(edges).id(d => d.id).distance(120))
            .force('charge', d3.forceManyBody().strength(-400))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collision', d3.forceCollide().radius(d => getRadius(d.entity_type) + 8))
            .on('tick', ticked);
    }

    function ticked() {
        linkElements
            .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
        linkLabelElements
            .attr('x', d => (d.source.x + d.target.x) / 2)
            .attr('y', d => (d.source.y + d.target.y) / 2);
        nodeElements.attr('cx', d => d.x).attr('cy', d => d.y);
        labelElements.attr('x', d => d.x).attr('y', d => d.y);
    }

    function dragStarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x; d.fy = d.y;
    }
    function dragged(event, d) { d.fx = event.x; d.fy = event.y; }
    function dragEnded(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null; d.fy = null;
    }

    async function nodeClicked(event, d) {
        event.stopPropagation();
        const detail = document.getElementById('node-detail');
        document.getElementById('node-name').textContent = d.name;
        document.getElementById('node-type').textContent = 'Type: ' + d.entity_type;
        document.getElementById('node-desc').textContent = d.description || 'No description';
        detail.style.display = 'block';

        // Load linked entries
        const linkedEl = document.getElementById('linked-entries');
        try {
            const resp = await fetch('/api/search?q=' + encodeURIComponent(d.name) + '&top_k=5');
            const data = await resp.json();
            const results = data.results || [];
            if (results.length) {
                linkedEl.innerHTML = '<div style="font-size:10px;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px">Related Memories</div>' +
                    results.map(r => `<div class="linked-entry">• ${truncate(r.l0 || 'No content', 60)}</div>`).join('');
            } else {
                linkedEl.innerHTML = '<div style="font-size:10px;color:#94a3b8">No related memories</div>';
            }
        } catch(e) {
            linkedEl.innerHTML = '';
        }
    }

    function truncate(str, len) { return str.length > len ? str.substring(0, len) + '...' : str; }

    function highlight(name) {
        const nameLower = name.toLowerCase();
        nodeElements.attr('opacity', d => d.name.toLowerCase().includes(nameLower) ? 1 : 0.15);
        labelElements.attr('opacity', d => d.name.toLowerCase().includes(nameLower) ? 1 : 0.15);
        linkElements.attr('opacity', d =>
            (d.source.name && d.source.name.toLowerCase().includes(nameLower)) ||
            (d.target.name && d.target.name.toLowerCase().includes(nameLower)) ? 0.8 : 0.05
        );
        linkLabelElements.attr('opacity', d =>
            (d.source.name && d.source.name.toLowerCase().includes(nameLower)) ||
            (d.target.name && d.target.name.toLowerCase().includes(nameLower)) ? 0.8 : 0.05
        );
    }

    function reset() {
        nodeElements.attr('opacity', 1);
        labelElements.attr('opacity', 1);
        linkElements.attr('opacity', 1);
        linkLabelElements.attr('opacity', 1);
        document.getElementById('node-detail').style.display = 'none';
        svg.transition().duration(500).call(
            d3.zoom().scaleExtent([0.2, 6]).on('zoom', (event) => {
                svg.select('g').attr('transform', event.transform);
            }).transform,
            d3.zoomIdentity
        );
    }

    return { init, highlight, reset, getColor };
})();
