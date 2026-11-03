"""Build contour-line hero maps from ETOPO 10-arc-minute elevation (NOAA, via pyvista/vtk-data)."""
import json, math, os, random, sys
import numpy as np
from scipy import ndimage
from skimage import measure

G = np.load(os.path.join(os.path.dirname(__file__), 'globe.npy'))  # rows: lat 90 -> -90, cols: lon -180 -> 180, 1/6 deg
OUT = sys.argv[1]
VW, ASPECT = 1209, 2.2
VH = round(VW / ASPECT)

# slug, caption, feature lat, lon, frame width in km, where the feature sits horizontally (0..1)
REGIONS = [
    ('granada',    'Sierra Nevada and Granada, Spain',       37.1,   -3.6,   900, .64, 1.0),
    ('geneva',     'Lake Geneva and the Alps',               46.2,    6.6,  1100, .40, 1.7),
    ('chongqing',  'Chongqing and the Sichuan Basin, China',  29.6,  106.5,  2100, .55, 2.1),
    ('kyoto',      'Kyoto and central Honshu, Japan',         35.0,  135.8,   800, .60, 1.0),
    ('hawaii',     "O'ahu and the Hawaiian Islands",          20.6, -157.3,   950, .58, 1.0),
    ('atlas',      'The Atlas Mountains, Morocco',            31.5,   -6.5,  1500, .62, 1.1),
    ('iceland',    'Iceland',                                 64.9,  -18.6,  1250, .66, 1.0),
    ('himalaya',   'Everest and the Himalaya',                28.0,   86.9,  1400, .60, 1.7),
    ('taiwan',     'Taiwan',                                  23.7,  121.0,   800, .62, 1.0),
    ('socal',      'The San Gabriel Mountains, California',   34.2, -117.7,   800, .62, 1.1),
    ('altiplano',  'Lake Titicaca and the Altiplano, Andes',  -16.0, -69.3,  1300, .62, 1.3),
    ('newzealand', 'The Southern Alps, New Zealand',         -43.6,  171.0,  1500, .55, 1.2),
    ('sicily',     'Mount Etna and Sicily, Italy',            37.6,   14.2,   700, .60, 1.0),
    ('srilanka',   'The Central Highlands, Sri Lanka',         7.7,   80.7,  1500, .64, 1.0),
    ('korea',      'The Korean Peninsula',                    37.5,  127.8,  1500, .60, 1.2),
    ('kamchatka',  'The volcanoes of Kamchatka, Russia',      55.5,  159.5,  1300, .60, 1.2),
    ('patagonia',  'Torres del Paine and Patagonia, Chile',  -50.0,  -73.5,  1100, .58, 1.2),
    ('cascades',   'Mount Rainier and Puget Sound, Washington', 46.9, -121.8,  700, .62, 1.1),
    ('sierra',     'Yosemite and the Sierra Nevada, California', 37.7, -119.5, 900, .62, 1.1),
    ('pyrenees',   'The Pyrenees, France and Spain',          42.7,    0.8,   900, .55, 1.4),
    ('caucasus',   'The Caucasus, between two seas',          42.5,   44.5,  1500, .52, 1.4),
    ('canaries',   'Mount Teide and the Canary Islands',      28.3,  -16.0,   700, .60, 1.0),
    ('hokkaido',   'Hokkaido, Japan',                         43.3,  142.8,   800, .62, 1.0),
    ('tasmania',   'Tasmania, Australia',                    -42.0,  146.6,   700, .62, 1.0),
]

def nice(x):
    for s in (25, 50, 100, 150, 200, 250, 300, 400, 500, 750, 1000):
        if x <= s:
            return s
    return 1000

def window(lat, lon, wkm, pos):
    hkm = wkm / ASPECT
    dlat = hkm / 111.0
    dlon = wkm / (111.0 * math.cos(math.radians(lat)))
    lon0 = lon - pos * dlon
    lat1 = lat + dlat / 2
    r0 = int(round((90 - lat1) * 6)); r1 = r0 + int(round(dlat * 6)) + 1
    c0 = int(round((lon0 + 180) * 6)); nc = int(round(dlon * 6)) + 1
    cols = [(c0 + i) % 2160 for i in range(nc)]
    return G[r0:r1][:, cols].astype(float)

def smooth_path(c, k):
    c = measure.approximate_polygon(c, 0.35) * k
    if len(c) < 5:
        return None
    closed = np.allclose(c[0], c[-1])
    if closed and max(np.ptp(c[:, 0]), np.ptp(c[:, 1])) < 22:
        return None
    m = [(c[i] + c[i + 1]) / 2 for i in range(len(c) - 1)]
    d = f'M{m[0][1]:.0f} {m[0][0]:.0f}'
    for i in range(1, len(m)):
        d += f'Q{c[i][1]:.0f} {c[i][0]:.0f} {m[i][1]:.0f} {m[i][0]:.0f}'
    if closed:
        d += f'Q{c[0][1]:.0f} {c[0][0]:.0f} {m[0][1]:.0f} {m[0][0]:.0f}'
    return d

def build(slug, lat, lon, wkm, pos, sm=1.0):
    W = window(lat, lon, wkm, pos)
    # resample to square cells, ~620 px wide, then smooth the 18 km grid steps away
    ky = 1.0; kx = math.cos(math.radians(lat))
    target = 620 / W.shape[1] / kx
    Z = ndimage.zoom(W, (target, target * kx), order=3)
    Z = ndimage.gaussian_filter(Z, max(1.0, target * 0.22) * sm)
    K = VW / Z.shape[1]
    land_max = max(Z.max(), 1)
    sea_min = min(Z.min(), 0)
    ls = nice(land_max / 17)
    ss = nice(-sea_min / 7) if sea_min < -50 else None
    groups = {'sea': [], 'land': [], 'ix': []}
    has_sea = (Z < 0).mean() > 0.02
    for lv in np.arange(ls, land_max, ls):
        idx = int(round(lv / ls))
        key = 'ix' if (not has_sea and idx % 4 == 0) else 'land'
        for c in measure.find_contours(Z, lv):
            if len(c) >= 8 and (p := smooth_path(c, K)):
                groups[key].append(p)
    if has_sea:
        for c in measure.find_contours(Z, 0):
            if len(c) >= 6 and (p := smooth_path(c, K)):
                groups['ix'].append(p)
        if ss:
            for lv in np.arange(-ss, sea_min, -ss):
                for c in measure.find_contours(Z, lv):
                    if len(c) >= 10 and (p := smooth_path(c, K)):
                        groups['sea'].append(p)
    rnd = random.Random(slug)
    P = lambda d: f'<path pathLength="1" style="--td:{rnd.uniform(0, .9):.2f}s" d="{d}"/>'
    parts = []
    for g in ('sea', 'land', 'ix'):
        if groups[g]:
            cls = '' if g == 'land' else f' class="{g}"'
            parts.append(f'<g{cls}>' + ''.join(P(d) for d in groups[g]) + '</g>')
    vh = round(Z.shape[0] * K)
    return ''.join(parts), vh, {g: len(v) for g, v in groups.items()}, (ls, ss)

if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    manifest = []
    for slug, cap, lat, lon, wkm, pos, sm in REGIONS:
        svg, vh, counts, steps = build(slug, lat, lon, wkm, pos, sm)
        with open(os.path.join(OUT, slug + '.svg'), 'w', encoding='utf-8') as f:
            f.write(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {VW} {vh}">{svg}</svg>')
        manifest.append({'id': slug, 'name': cap, 'vh': vh})
        print(f'{slug:12s} {len(svg)/1024:6.1f} KB  {counts}  steps={steps}')
    with open(os.path.join(OUT, 'maps.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False)
