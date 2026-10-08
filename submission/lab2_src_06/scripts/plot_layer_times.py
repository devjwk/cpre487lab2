#!/usr/bin/env python3
"""Draw the layer-wise execution time fraction from the test logs as an SVG bar chart.

Usage: python3 scripts/plot_layer_times.py [output.svg]

Reads results/x86_co2050_run.txt and results/zedboard_raw.txt (output of ./build/ml and scripts/flash_vitis).
The time of a layer is the mean of its "Layer Inference" timer over the three test images.
"""
import re
import sys

LAYERS = ['conv1', 'conv2', 'pool1', 'conv3', 'conv4', 'pool2', 'conv5', 'conv6', 'pool3', 'flatten', 'dense1', 'dense2+softmax']
SERIES = [('x86 (i7-12700)', 'results/x86_co2050_run.txt', '#1D4ED8'), ('ZedBoard', 'results/zedboard_raw.txt', '#93C5FD')]


def layer_times(path):
    """Mean time in ms of each layer test, in layer order."""
    text = open(path, errors='replace').read()
    times, name = {}, None
    for line in text.splitlines():
        m = re.search(r'--- Running Layer Test ([0-9 +]+), image', line)
        if m:
            name = m.group(1).strip()
        elif 'Running' in line:
            name = None
        elif name and 'Layer Inference: elapsed=' in line:
            times.setdefault(name, []).append(float(line.split('elapsed=')[1].split('ms')[0]))
    assert len(times) == len(LAYERS), f'{path}: expected {len(LAYERS)} layer tests, found {len(times)}'
    return [sum(v) / len(v) for v in times.values()]


def main(out):
    data = []
    for label, path, color in SERIES:
        ms = layer_times(path)
        data.append((label, color, [t / sum(ms) * 100 for t in ms], sum(ms)))

    W, H, left, top, plot_h = 1200, 460, 70, 50, 300
    step = (W - left - 30) / len(LAYERS)
    bar = step * 0.36
    font = "ui-sans-serif,system-ui,-apple-system,'Segoe UI',Helvetica,Arial,sans-serif"
    y = lambda pct: top + plot_h * (1 - pct / 70)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-label="Layer-wise execution time fraction">',
           '<title>Layer-wise execution time fraction</title>',
           f'<style>text{{font:500 13px {font};fill:#334155}}.t{{font:700 16px {font};fill:#0B1220}}.v{{font:500 11px {font}}}</style>',
           f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
           f'<text x="{left}" y="28" class="t">Layer-wise execution time fraction (naive C++, one image)</text>']
    for pct in range(0, 71, 10):
        svg.append(f'<line x1="{left}" y1="{y(pct):.1f}" x2="{W - 30}" y2="{y(pct):.1f}" stroke="#E2E8F0"/>')
        svg.append(f'<text x="{left - 8}" y="{y(pct) + 4:.1f}" text-anchor="end">{pct}%</text>')
    for i, name in enumerate(LAYERS):
        x0 = left + i * step + step * 0.12
        for s, (label, color, share, total) in enumerate(data):
            x = x0 + s * (bar + 2)
            svg.append(f'<rect x="{x:.1f}" y="{y(share[i]):.1f}" width="{bar:.1f}" height="{y(0) - y(share[i]):.1f}" fill="{color}"/>')
            svg.append(f'<text x="{x + bar / 2:.1f}" y="{y(share[i]) - 4:.1f}" text-anchor="middle" class="v">{share[i]:.1f}</text>')
        svg.append(f'<text x="{x0 + bar + 1:.1f}" y="{y(0) + 20:.1f}" text-anchor="middle">{name}</text>')
    for s, (label, color, share, total) in enumerate(data):
        lx = left + 20 + s * 330
        svg.append(f'<rect x="{lx}" y="{H - 40}" width="14" height="14" fill="{color}"/>')
        svg.append(f'<text x="{lx + 22}" y="{H - 28}">{label}: {total:,.1f} ms for all layers</text>')
    svg.append(f'<text x="{left}" y="{H - 8}" class="v">Share of the summed layer times, in percent. ZedBoard timer resolution is 1 ms, so layers under 1 ms read as 0.</text>')
    open(out, 'w').write('\n'.join(svg + ['</svg>']) + '\n')
    for label, color, share, total in data:
        print(label, f'{total:.1f} ms', ' '.join(f'{v:.1f}' for v in share))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'assets/layer_time_share.svg')
