import math
import colorsys
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from collections import defaultdict
from pyscript import document, display


class Tabloid:
    def __init__(self, rows):
        # rows: list of lists of integers
        self.rows = rows

    def display(self):
        for row in self.rows:
            # print entries without brackets or commas
            print(' '.join(str(x) for x in row))

    def __repr__(self):
        # for debugging
        return f"Tabloid({self.rows})"

class GrowthDiagram:
    def __init__(self, L):
        self.L = L
        self.n = len(L)
        # check affine permutation
        mods = [a % self.n for a in L]
        if len(set(mods)) != self.n:
            raise ValueError("Input is not an affine permutation modulo n.")
        # compute B, T, and dimensions
        self.x, self.y = min(L), max(L)
        self.B = ((self.x-1) // self.n) * self.n ## fixing
        self.T = ((self.y + self.n - 1) // self.n + 1) * self.n
        self.num_rows = self.T - self.B
        if self.num_rows % self.n != 0:
            raise ValueError("Total rows (T−B) must be a multiple of n.")
        # initialize structures
        self._place_dots()
        self.compute_edge_labels()
        self._compute_segments()
        self._compute_components()
        self._build_boundary_map()
        self.seg_color = {}
        self.color_list = []

    def _place_dots(self):
        # y-coordinate is a - self.B (0-indexed row relative to B)
        self.dots = [(i+1, a - self.B) for i, a in enumerate(self.L)]

    def compute_edge_labels(self):
        H = [[None] * self.n for _ in range(self.num_rows+1)]
        V = [[None] * (self.n+1) for _ in range(self.num_rows)]
        windows = self.num_rows // self.n
        for w in range(windows):
            r0, r1 = w * self.n, w * self.n + self.n - 1
            if w == 0:
                for c in range(self.n): H[0][c] = 0
                for r in range(r0, r1+1): V[r][self.n] = 0
            else:
                for r in range(r0, r1+1): V[r][self.n] = V[r-self.n][0]
            for r in range(r0, r1+1):
                for c in range(self.n-1, -1, -1):
                    top, right = H[r][c], V[r][c+1]
                    has_dot = ((c+1, r+1) in self.dots)
                    if top == 0 and right == 0:
                        bottom = left = 1 if has_dot else 0
                    else:
                        if top == right:
                            bottom = left = top + 1
                        else:
                            bottom = top
                            left = right
                    H[r+1][c] = bottom
                    V[r][c] = left
        self.H, self.V = H, V

    def _compute_segments(self):
        segs = []
        for r in range(self.num_rows):
            for c in range(self.n):
                top, right = self.H[r][c], self.V[r][c+1]
                bottom, left = self.H[r+1][c], self.V[r][c]
                # case 2: quarter-circle
                if top==0 and right==0 and bottom==1 and left==1:
                    ep = ((c, r+0.5), (c+0.5, r+1))
                    segs.append({'type':'arc', 'center':(c,r+1), 'theta1':270, 'theta2':360,
                                 'endpoints':ep, 'label':1})
                # case 4: two arcs
                elif top==right and left==bottom and top!=0:
                    ep1 = ((c+0.5, r), (c+1, r+0.5))
                    segs.append({'type':'arc', 'center':(c+1,r), 'theta1':90, 'theta2':180,
                                 'endpoints':ep1, 'label':top})
                    ep2 = ((c, r+0.5), (c+0.5, r+1))
                    segs.append({'type':'arc', 'center':(c,r+1), 'theta1':270, 'theta2':360,
                                 'endpoints':ep2, 'label':left})
                else:
                    if top==bottom and top!=0:
                        ep = ((c+0.5, r), (c+0.5, r+1))
                        segs.append({'type':'line', 'points':ep, 'endpoints':ep, 'label':top})
                    if left==right and left!=0:
                        ep = ((c, r+0.5), (c+1, r+0.5))
                        segs.append({'type':'line', 'points':ep, 'endpoints':ep, 'label':left})
        self.segments = segs

    def _compute_components(self):
        adj = defaultdict(set)
        for i, s1 in enumerate(self.segments):
            e1 = set(s1['endpoints'])
            for j, s2 in enumerate(self.segments):
                if i<j and e1 & set(s2['endpoints']):
                    adj[i].add(j); adj[j].add(i)
        seen, comps, comp_id = set(), [], {}
        for i in range(len(self.segments)):
            if i in seen: continue
            stack, comp = [i], set()
            while stack:
                u = stack.pop()
                if u in comp: continue
                comp.add(u); seen.add(u)
                for v in adj[u]: stack.append(v)
            cid = len(comps)
            for idx in comp: comp_id[idx] = cid
            comps.append(comp)
        self.adj, self.comps, self.comp_id = adj, comps, comp_id

    def _build_boundary_map(self):
        bm = defaultdict(list)
        for idx, seg in enumerate(self.segments):
            for x,y in seg['endpoints']:
                if abs(x)<1e-6:
                    bm[seg['label']].append(idx)
                    break
        self.boundary = bm

    def auto_color_left_sequence(self):
        C = len(self.boundary.get(1, []))
        if C==0: return
        sat, val = 0.7, 0.9
        self.color_list = [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" \
            for r,g,b in [colorsys.hsv_to_rgb(i/C, sat, val) for i in range(C)]]
        self.seg_color = {}
        for i, color in enumerate(self.color_list):
            x = 1
            while True:
                boundary_segs = self.boundary.get(x, [])
                seg_idx = next((s for s in boundary_segs if s not in self.seg_color), None)
                if seg_idx is None: break
                cid = self.comp_id[seg_idx]
                for s_idx in self.comps[cid]:
                    self.seg_color[s_idx] = color
                x += 1
            # x is implicitly reset to 1 at the start of the outer loop for the next color.

    def display_colored_shadow(self, scale=1.0):
        """
        Draw the full growth diagram with colored shadow lines.
        Automatically colors if not already colored.
        """
        if not self.seg_color:
            self.auto_color_left_sequence()
        
        # Adjust figsize: make it less aggressively large, aspect ratio can be handled by set_aspect and xlim/ylim
        # The goal is to produce a plot with minimal internal padding.
        fig_width = 7 * scale 
        fig_height = (self.num_rows / self.n) * fig_width * scale if self.n > 0 else fig_width # Attempt to match aspect ratio
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.set_aspect('equal')
        # grid & windows
        for x in range(self.n+1): ax.plot([x,x],[0,self.num_rows],'lightgray',lw=0.7)
        for y in range(self.num_rows+1): ax.plot([0,self.n],[y,y],'lightgray',lw=0.7)
        for k in range(1, self.num_rows//self.n): y=k*self.n; ax.plot([0,self.n],[y,y],'black',lw=2)
        ax.plot([0,self.n,self.n,0,0],[0,0,self.num_rows,self.num_rows,0],'black',lw=2)
        # dots
        xs = [c-0.5 for c,r in self.dots]
        ys = [r-0.5 for c,r in self.dots]
        ax.scatter(xs, ys, s=30*scale, color='blue', zorder=3)
        # segments
        for idx, seg in enumerate(self.segments):
            col = self.seg_color.get(idx, 'black')
            lw = 2 if idx in self.seg_color else 1
            if seg['type']=='line':
                (x1,y1),(x2,y2) = seg['points']
                ax.plot([x1,x2],[y1,y2], color=col, lw=lw, zorder=2)
            else:
                cx,cy = seg['center']
                arc = Arc((cx, cy),1,1, theta1=seg['theta1'], theta2=seg['theta2'], color=col, lw=lw, zorder=2)
                ax.add_patch(arc)
        
        ax.set_xlim(-0.5, self.n + 0.5) # Try to set x-limits tightly
        ax.set_ylim(self.num_rows + 0.5, -0.5) # Try to set y-limits tightly (inverted y-axis)
        ax.invert_yaxis() # Ensure y-axis is inverted
        ax.axis('off')
        fig.tight_layout(pad=0) 
        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01) # Squeeze margins further
        return fig

    def display_edge_labels_matplotlib(self, scale=1.0): # Renamed to avoid conflict
        """
        Draw a dotted growth diagram grid, show uncolored shadow lines in gray, 
        and annotate each edge label.
        Returns the Matplotlib figure.
        """
        fig_width = 7 * scale
        fig_height = (self.num_rows / self.n) * fig_width * scale if self.n > 0 else fig_width
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        ax.set_aspect('equal')
        # draw dotted grid lines
        for x_coord in range(self.n + 1):
            ax.plot([x_coord, x_coord], [0, self.num_rows], linestyle=':', color='darkgray', lw=0.5, zorder=0)
        for y_coord in range(self.num_rows + 1):
            ax.plot([0, self.n], [y_coord, y_coord], linestyle=':', color='darkgray', lw=0.5, zorder=0)
        
        # draw shadow lines (uncolored) in dashed gray
        for seg in self.segments:
            if seg['type'] == 'line':
                (x1, y1), (x2, y2) = seg['endpoints']
                ax.plot([x1, x2], [y1, y2], color='gray', lw=1, linestyle='--', zorder=1)
            else:
                cx, cy = seg['center']
                arc = Arc((cx, cy), 1, 1,
                          theta1=seg['theta1'], theta2=seg['theta2'],
                          color='gray', lw=1, linestyle='--', zorder=1)
                ax.add_patch(arc)
        # annotate horizontal edges
        for r_idx in range(self.num_rows + 1):
            for c_idx in range(self.n):
                label = self.H[r_idx][c_idx]
                ax.text(c_idx + 0.5, r_idx, str(label), ha='center', va='center', fontsize=8*scale, color='black', zorder=2)
        # annotate vertical edges
        for r_idx in range(self.num_rows):
            for c_idx in range(self.n + 1):
                label = self.V[r_idx][c_idx]
                ax.text(c_idx, r_idx + 0.5, str(label), ha='center', va='center', fontsize=8*scale, color='black', zorder=2)
        # plot dots (ensure dot coordinates are correct for plotting at cell centers)
        xs = [c - 0.5 for c, r in self.dots] # If self.dots stores 1-indexed cell coords
        ys = [r - 0.5 for c, r in self.dots] # If self.dots stores 1-indexed cell coords
        ax.scatter(xs, ys, s=30*scale, color='blue', zorder=3)
        
        ax.set_xlim(-0.5, self.n + 0.5) # Try to set x-limits tightly
        ax.set_ylim(self.num_rows + 0.5, -0.5) # Try to set y-limits tightly (inverted y-axis)
        ax.invert_yaxis()
        ax.axis('off')
        fig.tight_layout(pad=0)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01) # Squeeze margins further
        return fig

    def display_colored_shadow_html(self):
        svg_parts = []
        
        view_box_width = self.n 
        view_box_height = self.num_rows
        if view_box_width == 0 or view_box_height == 0: # Avoid zero viewBox dimensions
            return "<svg width='100%' height='100px'><text x='10' y='20'>Error: Diagram dimensions are zero.</text></svg>"

        line_stroke_width = 0.02
        colored_line_stroke_width = 0.05
        dot_radius = 0.08

        svg_parts.append(f'<svg viewBox="-0.5 -0.5 {view_box_width+1} {view_box_height+1}" xmlns="http://www.w3.org/2000/svg" style="width:100%; height:auto; background-color: white;">')

        if not self.seg_color:
            self.auto_color_left_sequence()

        for x_coord in range(self.n + 1):
            svg_parts.append(f'<line x1="{x_coord}" y1="0" x2="{x_coord}" y2="{self.num_rows}" stroke="lightgray" stroke-width="{line_stroke_width}" />')
        for y_coord in range(self.num_rows + 1):
            svg_parts.append(f'<line x1="0" y1="{y_coord}" x2="{self.n}" y2="{y_coord}" stroke="lightgray" stroke-width="{line_stroke_width}" />')
        
        for k in range(1, self.num_rows // self.n):
            y = k * self.n
            svg_parts.append(f'<line x1="0" y1="{y}" x2="{self.n}" y2="{y}" stroke="black" stroke-width="{line_stroke_width * 2}" />')
        svg_parts.append(f'<rect x="0" y="0" width="{self.n}" height="{self.num_rows}" fill="none" stroke="black" stroke-width="{line_stroke_width * 2}" />')

        for idx, seg in enumerate(self.segments):
            color = self.seg_color.get(idx, 'gray') 
            stroke_w = colored_line_stroke_width if idx in self.seg_color else line_stroke_width * 1.5
            
            if seg['type'] == 'line':
                (x1, y1), (x2, y2) = seg['points']
                svg_parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{stroke_w}" />')
            else: 
                cx, cy = seg['center']
                radius = 0.5 
                theta1_deg, theta2_deg = seg['theta1'], seg['theta2']
                t1_rad, t2_rad = math.radians(theta1_deg), math.radians(theta2_deg)
                sx, sy = cx + radius * math.cos(t1_rad), cy + radius * math.sin(t1_rad)
                ex, ey = cx + radius * math.cos(t2_rad), cy + radius * math.sin(t2_rad)
                svg_parts.append(f'<path d="M {sx:.4f},{sy:.4f} A {radius:.4f},{radius:.4f} 0 0 1 {ex:.4f},{ey:.4f}" stroke="{color}" stroke-width="{stroke_w}" fill="none" />')

        for c_dot, r_dot in self.dots:
            cx_svg, cy_svg = c_dot - 0.5, r_dot - 0.5
            svg_parts.append(f'<circle cx="{cx_svg:.4f}" cy="{cy_svg:.4f}" r="{dot_radius:.4f}" fill="blue" />')
            
        svg_parts.append('</svg>')
        return "".join(svg_parts)

    def display_edge_labels_html(self):
        svg_parts = []
        view_box_width = self.n
        view_box_height = self.num_rows
        if view_box_width == 0 or view_box_height == 0:
            return "<svg width='100%' height='100px'><text x='10' y='20'>Error: Diagram dimensions are zero.</text></svg>"


        line_stroke_width = 0.02
        label_font_size = 0.3 
        dot_radius = 0.08

        svg_parts.append(f'<svg viewBox="-0.5 -0.5 {view_box_width+1} {view_box_height+1}" xmlns="http://www.w3.org/2000/svg" style="width:100%; height:auto; background-color: white;">')

        for x_coord in range(self.n + 1):
            svg_parts.append(f'<line x1="{x_coord}" y1="0" x2="{x_coord}" y2="{self.num_rows}" stroke="darkgray" stroke-width="{line_stroke_width}" stroke-dasharray="{line_stroke_width*2},{line_stroke_width*2}" />')
        for y_coord in range(self.num_rows + 1):
            svg_parts.append(f'<line x1="0" y1="{y_coord}" x2="{self.n}" y2="{y_coord}" stroke="darkgray" stroke-width="{line_stroke_width}" stroke-dasharray="{line_stroke_width*2},{line_stroke_width*2}" />')

        for seg in self.segments:
            stroke_w = line_stroke_width * 1.5
            if seg['type'] == 'line':
                (x1, y1), (x2, y2) = seg['endpoints']
                svg_parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="gray" stroke-width="{stroke_w}" stroke-dasharray="{line_stroke_width*4},{line_stroke_width*4}" />')
            else: 
                cx, cy = seg['center']
                radius = 0.5
                theta1_deg, theta2_deg = seg['theta1'], seg['theta2']
                t1_rad, t2_rad = math.radians(theta1_deg), math.radians(theta2_deg)
                sx, sy = cx + radius * math.cos(t1_rad), cy + radius * math.sin(t1_rad)
                ex, ey = cx + radius * math.cos(t2_rad), cy + radius * math.sin(t2_rad)
                svg_parts.append(f'<path d="M {sx:.4f},{sy:.4f} A {radius:.4f},{radius:.4f} 0 0 1 {ex:.4f},{ey:.4f}" stroke="gray" stroke-width="{stroke_w}" fill="none" stroke-dasharray="{line_stroke_width*4},{line_stroke_width*4}" />')

        for r_idx in range(self.num_rows + 1):
            for c_idx in range(self.n):
                label = self.H[r_idx][c_idx]
                svg_parts.append(f'<text x="{c_idx + 0.5}" y="{r_idx}" font-size="{label_font_size}" text-anchor="middle" dominant-baseline="middle" fill="black">{label}</text>')
        
        for r_idx in range(self.num_rows):
            for c_idx in range(self.n + 1):
                label = self.V[r_idx][c_idx]
                svg_parts.append(f'<text x="{c_idx}" y="{r_idx + 0.5}" font-size="{label_font_size}" text-anchor="middle" dominant-baseline="middle" fill="black">{label}</text>')

        for c_dot, r_dot in self.dots:
            cx_svg, cy_svg = c_dot - 0.5, r_dot - 0.5
            svg_parts.append(f'<circle cx="{cx_svg:.4f}" cy="{cy_svg:.4f}" r="{dot_radius:.4f}" fill="blue" />')

        svg_parts.append('</svg>')
        return "".join(svg_parts)

    def extract_final_window(self):
        n=self.n; r0=(self.num_rows//n-1)*n; r1=r0+n
        Hw = [row[:] for row in self.H[r0:r1+1]]
        Vw = [row[:] for row in self.V[r0:r1]]
        dots = [(c, r-r0) for c,r in self.dots if r0<r<=r1]
        segs, cols = [], {}
        for old_idx, seg in enumerate(self.segments):
            ys = [y for x,y in seg['endpoints']]
            if all(r0<=y<=r1 for y in ys):
                new = seg.copy()
                new['endpoints'] = tuple((x, y-r0) for x,y in seg['endpoints'])
                if new['type']=='line':
                    new['points'] = new['endpoints']
                else:
                    cx, cy = new['center']; new['center'] = (cx, cy-r0)
                idx_new = len(segs)
                segs.append(new)
                if old_idx in self.seg_color:
                    cols[idx_new] = self.seg_color[old_idx]
        return {'H':Hw, 'V':Vw, 'dots':dots, 'segments':segs, 'seg_color':cols}

    def display_final_window(self, scale=1.0):
        data = self.extract_final_window(); n=self.n
        fig, ax = plt.subplots(figsize=(6*scale,6*scale)); ax.set_aspect('equal')
        for x in range(n+1): ax.plot([x,x],[0,n],'lightgray',lw=0.7)
        for y in range(n+1): ax.plot([0,n],[y,y],'lightgray',lw=0.7)
        xs = [c-0.5 for c,r in data['dots']]; ys=[r-0.5 for c,r in data['dots']]
        ax.scatter(xs, ys, s=30*scale, color='blue', zorder=3)
        for idx, seg in enumerate(data['segments']):
            col = data['seg_color'].get(idx, 'black'); lw = 2 if idx in data['seg_color'] else 1
            if seg['type']=='line':
                (x1,y1),(x2,y2) = seg['points']; ax.plot([x1,x2],[y1,y2], color=col, lw=lw, zorder=2)
            else:
                cx, cy = seg['center']
                arc = Arc((cx, cy),1,1, theta1=seg['theta1'], theta2=seg['theta2'], color=col, lw=lw, zorder=2)
                ax.add_patch(arc)
        ax.invert_yaxis(); ax.axis('off'); plt.show()

    def P_tabloid_list(self):
        # The call to display_colored_shadow in run_app ensures auto_color_left_sequence is attempted.
        # If color_list or seg_color are empty after that, it's a valid state (e.g., C=0).
        data = self.extract_final_window(); segs,cols = data['segments'], data['seg_color']
        result = []
        for row in range(1, self.n+1):
            mid = (0, row-0.5)
            idx = next((i for i,s in enumerate(segs) if any(abs(x-mid[0])<1e-6 and abs(y-mid[1])<1e-6 for x,y in s['endpoints'])), None)
            if idx is None:
                result.append(None)
            else:
                cstr = cols.get(idx)
                result.append(self.color_list.index(cstr)+1 if cstr in self.color_list else None)
        return result

    def Q_tabloid_list(self):
        if not self.color_list or not self.seg_color:
            raise ValueError("Call auto_color_left_sequence first.")
        data = self.extract_final_window(); segs,cols = data['segments'], data['seg_color']
        result = []
        for col in range(1, self.n+1):
            mid = (col-0.5, self.n)
            idx = next((i for i,s in enumerate(segs) if any(abs(x-mid[0])<1e-6 and abs(y-mid[1])<1e-6 for x,y in s['endpoints'])), None)
            if idx is None:
                result.append(None)
            else:
                cstr = cols.get(idx)
                result.append(self.color_list.index(cstr)+1 if cstr in self.color_list else None)
        return result

    def list_2_tabloid(self, lst):
        maxl = max((v for v in lst if v), default=0)
        return [[i+1 for i,v in enumerate(lst) if v==k] for k in range(1, maxl+1)]

    def P_tabloid(self):
        rows = self.list_2_tabloid(self.P_tabloid_list())
        return Tabloid(rows)

    def Q_tabloid(self):
        rows = self.list_2_tabloid(self.Q_tabloid_list())##
        return Tabloid(rows[::-1])

    def display_P_tabloid(self):
        print("P-tabloid:")
        self.P_tabloid().display()

    def display_Q_tabloid(self):
        print("Q-tabloid:")
        self.Q_tabloid().display()

    def lambda_list(self):
        # The call to display_colored_shadow in run_app ensures auto_color_left_sequence is attempted.
        # If color_list or seg_color are empty after that, it's a valid state (e.g., C=0).
        L_val_internal = [] # Renamed from L to avoid confusion with self.L or input L
        for row in range(1, self.num_rows+1):
            mid = (self.n, row-0.5)
            idx = next((i for i, s in enumerate(self.segments)
                        if any(abs(x-mid[0])<1e-6 and abs(y-mid[1])<1e-6 for x,y in s['endpoints'])), None)
            if idx is None:
                L_val_internal.append(None)
            else:
                cstr = self.seg_color.get(idx)
                L_val_internal.append(self.color_list.index(cstr)+1 if cstr in self.color_list else None)
        L_trunc = L_val_internal[:-self.n]
        maxlab = max((v for v in L_trunc if v), default=0)
        return [L_trunc.count(i) for i in range(1, maxlab+1)]

    def affine_RS(self, display_output=True): # Renamed display to display_output for clarity
        if not self.seg_color and not self.color_list: # Ensure coloring is done if needed
            self.auto_color_left_sequence()
        P = self.P_tabloid()
        Q = self.Q_tabloid()
        lam = self.lambda_list()
        if display_output: # Only print if called directly, not from web app
            self.display_P_tabloid()
            self.display_Q_tabloid()
        return P, Q, lam

def run_app(evt=None):
    # diag_elem = document.getElementById('diagram') # Old, not used
    res_elem = document.getElementById('result')

    # Clear previous output
    diag_output_elem = document.getElementById('diagram-output-area')
    diag_output_elem.innerHTML = '' # Clear previous plots
    res_elem.textContent = ''

    try:
        perm_input_value = document.getElementById('perm').value.strip() # Renamed for clarity
        if not perm_input_value:
            res_elem.textContent = "Please enter an affine permutation."
            return

        L = [int(x) for x in perm_input_value.split(',') if x.strip()]

        w_display_string = f'w = [{perm_input_value}]\n\n' # Format the string to display w

        gd = GrowthDiagram(L)

        display_edge_labels_toggled = document.getElementById('toggleEdgeLabels').checked

        if display_edge_labels_toggled:
            diag_output_elem.innerHTML = '''
                <div class="row">
                    <div class="col-md-6" id="edge-labels-plot-target"></div>
                    <div class="col-md-6" id="colored-shadow-plot-target"></div>
                </div>
            '''
            svg_el = gd.display_edge_labels_html() 
            document.getElementById('edge-labels-plot-target').innerHTML = svg_el
            
            svg_cs = gd.display_colored_shadow_html()
            document.getElementById('colored-shadow-plot-target').innerHTML = svg_cs
        else:
            # Create a single target div for the plot if it doesn't exist or structure differently
            diag_output_elem.innerHTML = '<div id="colored-shadow-plot-target-single"></div>'
            svg_cs = gd.display_colored_shadow_html()
            document.getElementById('colored-shadow-plot-target-single').innerHTML = svg_cs

        # No need for plt.close('all') anymore as we are not using Matplotlib for display here

        # Calculate P, Q, lambda
        P_tab = gd.P_tabloid()
        Q_tab = gd.Q_tabloid()
        lam_list = gd.lambda_list()

        # Format and display P, Q, lambda
        out_P_rows = [' '.join(map(str, row)) for row in P_tab.rows] if P_tab.rows else ["(empty)"]
        out_Q_rows = [' '.join(map(str, row)) for row in Q_tab.rows] if Q_tab.rows else ["(empty)"]

        output_text = w_display_string + "P-tabloid:\n" + "\n".join(out_P_rows) + \
                      "\n\nQ-tabloid:\n" + "\n".join(out_Q_rows) + \
                      f"\n\nλ:\n{lam_list}"
        res_elem.textContent = output_text

    except ValueError as e:
        res_elem.textContent = f"Input Error: {e}"
    except Exception as e:
        res_elem.textContent = f"An unexpected error occurred: {e}\nCheck console for details."
        print(f"Full error: {type(e).__name__}: {e}") # Log full error to console for debugging