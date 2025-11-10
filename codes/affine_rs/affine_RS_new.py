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
        if self.n == 0:
            raise ValueError("Input permutation L cannot be empty.")

        mods = [a % self.n for a in L]
        if len(set(mods)) != self.n:
            raise ValueError("Input is not an affine permutation modulo n.")
        
        # compute B, T, and dimensions
        # L is non-empty here, so min(L) and max(L) are safe.
        self.x, self.y = min(L), max(L)
        self.B = ((self.x - 1) // self.n) * self.n
        self.T_initial = ((self.y + self.n - 1) // self.n + 1) * self.n # Store initial T
        self.num_rows = self.T_initial - self.B
        self.T = self.T_initial # Current T, will be updated if diagram extends        
        
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
        self.dots = [(i + 1, a - self.B) for i, a in enumerate(self.L)]

    def compute_edge_labels(self):
        H = [[None] * self.n for _ in range(self.num_rows + 1)]
        V = [[None] * (self.n + 1) for _ in range(self.num_rows)]
        windows = self.num_rows // self.n
        for w in range(windows):
            r0, r1 = w * self.n, w * self.n + self.n - 1
            if w == 0:
                for c in range(self.n): H[0][c] = 0
                for r in range(r0, r1 + 1): V[r][self.n] = 0
            else:
                for r in range(r0, r1 + 1): V[r][self.n] = V[r - self.n][0]
            for r in range(r0, r1 + 1):
                for c in range(self.n - 1, -1, -1):
                    top, right = H[r][c], V[r][c + 1]
                    has_dot = ((c + 1, r + 1) in self.dots)
                    if top == 0 and right == 0:
                        bottom = left = 1 if has_dot else 0
                    else:
                        if top == right:
                            bottom = left = top + 1
                        else:
                            bottom = top
                            left = right
                    H[r + 1][c] = bottom
                    V[r][c] = left
        self.H, self.V = H, V

    def _compute_segments(self):
        segs = []
        for r in range(self.num_rows):
            for c in range(self.n):
                top, right = self.H[r][c], self.V[r][c + 1]
                bottom, left = self.H[r + 1][c], self.V[r][c]
                # case 2: quarter-circle
                if top == 0 and right == 0 and bottom == 1 and left == 1:
                    ep = ((c, r + 0.5), (c + 0.5, r + 1))
                    segs.append({'type': 'arc', 'center': (c, r + 1), 'theta1': 270, 'theta2': 360,
                                 'endpoints': ep, 'label': 1})
                # case 4: two arcs
                elif top == right and left == bottom and top != 0:
                    ep1 = ((c + 0.5, r), (c + 1, r + 0.5))
                    segs.append({'type': 'arc', 'center': (c + 1, r), 'theta1': 90, 'theta2': 180,
                                 'endpoints': ep1, 'label': top})
                    ep2 = ((c, r + 0.5), (c + 0.5, r + 1))
                    segs.append({'type': 'arc', 'center': (c, r + 1), 'theta1': 270, 'theta2': 360,
                                 'endpoints': ep2, 'label': left})
                else:
                    if top == bottom and top != 0:
                        ep = ((c + 0.5, r), (c + 0.5, r + 1))
                        segs.append({'type': 'line', 'points': ep, 'endpoints': ep, 'label': top})
                    if left == right and left != 0:
                        ep = ((c, r + 0.5), (c + 1, r + 0.5))
                        segs.append({'type': 'line', 'points': ep, 'endpoints': ep, 'label': left})
        self.segments = segs

    def _compute_components(self):
        adj = defaultdict(set)
        for i, s1 in enumerate(self.segments):
            e1 = set(s1['endpoints'])
            for j, s2 in enumerate(self.segments):
                if i < j and e1 & set(s2['endpoints']):
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
            for x, y in seg['endpoints']:
                if abs(x) < 1e-6: # Check if x-coordinate is close to 0 (left boundary)
                    bm[seg['label']].append(idx)
                    break
        self.boundary = bm

    def auto_color_left_sequence(self):
        # Identify unique components that touch the left boundary and 
        # their lowest y-coordinate on that boundary.
        # This dictionary will store: comp_id -> min_y_coord_on_boundary
        components_on_boundary_info = {} 
        
        # Iterate over all labels present in self.boundary and all their respective segments.
        # self.boundary is a dict: label -> list of segment indices on boundary with that label.
        for _label, segment_indices_for_this_label in self.boundary.items():
            for seg_idx in segment_indices_for_this_label:
                # Basic safety check, though _build_boundary_map should ensure valid indices.
                if seg_idx >= len(self.segments): continue 
                
                comp_id = self.comp_id.get(seg_idx)
                # This should not happen if _compute_components is correct and seg_idx is valid.
                if comp_id is None: continue 

                # For this segment (which is on the boundary), find its minimum y-coordinate at x=0.
                min_y_for_this_segment_at_boundary = float('inf')
                for x_ep, y_ep in self.segments[seg_idx]['endpoints']:
                    if abs(x_ep) < 1e-6: # This endpoint is on the left boundary (x=0).
                        min_y_for_this_segment_at_boundary = min(min_y_for_this_segment_at_boundary, y_ep)
                
                # If segment from self.boundary didn't yield an x=0 endpoint y-coord (highly unlikely).
                if min_y_for_this_segment_at_boundary == float('inf'):
                    continue

                # Update the component's overall minimum y-contact point.
                if comp_id not in components_on_boundary_info:
                    components_on_boundary_info[comp_id] = min_y_for_this_segment_at_boundary
                else:
                    components_on_boundary_info[comp_id] = min(components_on_boundary_info[comp_id], 
                                                               min_y_for_this_segment_at_boundary)
        
        # Sort component IDs: primary key is min_y_coord, secondary key is comp_id (for stable sort).
        sorted_comp_ids = sorted(components_on_boundary_info.keys(),
                                 key=lambda cid: (components_on_boundary_info[cid], cid))

        C = len(sorted_comp_ids) # Number of distinct "strands" or components to color

        if C == 0:
            self.color_list = []
            self.seg_color = {}
            return

        sat, val = 0.7, 0.9
        self.color_list = [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" \
            for r,g,b in [colorsys.hsv_to_rgb(i/C, sat, val) for i in range(C)]]
        
        self.seg_color = {} # Clear previous coloring
        
        for i, comp_id_to_color in enumerate(sorted_comp_ids):
            color_to_assign = self.color_list[i]
            for seg_idx_in_comp in self.comps[comp_id_to_color]:
                self.seg_color[seg_idx_in_comp] = color_to_assign

    def display_colored_shadow(self, scale=1.0):
        """
        Draw the full growth diagram with colored shadow lines.
        Automatically colors if not already colored.
        """
        if not self.seg_color and not self.color_list: # Check if coloring is needed
            self.auto_color_left_sequence()
        
        fig_width = 7 * scale 
        fig_height = (self.num_rows / self.n) * fig_width * scale if self.n > 0 else fig_width
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.set_aspect('equal')
        
        for x in range(self.n + 1): ax.plot([x, x], [0, self.num_rows], 'lightgray', lw=0.7)
        for y in range(self.num_rows + 1): ax.plot([0, self.n], [y, y], 'lightgray', lw=0.7)
        for k in range(1, self.num_rows // self.n): 
            y = k * self.n
            ax.plot([0, self.n], [y, y], 'black', lw=2)
        ax.plot([0, self.n, self.n, 0, 0], [0, 0, self.num_rows, self.num_rows, 0], 'black', lw=2)
        
        xs = [c - 0.5 for c, r in self.dots]
        ys = [r - 0.5 for c, r in self.dots]
        ax.scatter(xs, ys, s=30 * scale, color='blue', zorder=3)
        
        for idx, seg in enumerate(self.segments):
            col = self.seg_color.get(idx, 'black')
            lw = 2 if idx in self.seg_color else 1
            if seg['type'] == 'line':
                (x1, y1), (x2, y2) = seg['points']
                ax.plot([x1, x2], [y1, y2], color=col, lw=lw, zorder=2)
            else:
                cx, cy = seg['center']
                arc = Arc((cx, cy), 1, 1, theta1=seg['theta1'], theta2=seg['theta2'], color=col, lw=lw, zorder=2)
                ax.add_patch(arc)
        
        ax.set_xlim(-0.5, self.n + 0.5)
        ax.set_ylim(self.num_rows + 0.5, -0.5)
        ax.invert_yaxis()
        ax.axis('off')
        fig.tight_layout(pad=0) 
        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        return fig

    def display_edge_labels_matplotlib(self, scale=1.0):
        fig_width = 7 * scale
        fig_height = (self.num_rows / self.n) * fig_width * scale if self.n > 0 else fig_width
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.set_aspect('equal')

        for x_coord in range(self.n + 1):
            ax.plot([x_coord, x_coord], [0, self.num_rows], linestyle=':', color='darkgray', lw=0.5, zorder=0)
        for y_coord in range(self.num_rows + 1):
            ax.plot([0, self.n], [y_coord, y_coord], linestyle=':', color='darkgray', lw=0.5, zorder=0)
        
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

        for r_idx in range(self.num_rows + 1):
            for c_idx in range(self.n):
                label = self.H[r_idx][c_idx]
                ax.text(c_idx + 0.5, r_idx, str(label), ha='center', va='center', fontsize=8*scale, color='black', zorder=2)
        
        for r_idx in range(self.num_rows):
            for c_idx in range(self.n + 1):
                label = self.V[r_idx][c_idx]
                ax.text(c_idx, r_idx + 0.5, str(label), ha='center', va='center', fontsize=8*scale, color='black', zorder=2)
        
        xs = [c - 0.5 for c, r in self.dots]
        ys = [r - 0.5 for c, r in self.dots]
        ax.scatter(xs, ys, s=30*scale, color='blue', zorder=3)
        
        ax.set_xlim(-0.5, self.n + 0.5)
        ax.set_ylim(self.num_rows + 0.5, -0.5)
        ax.invert_yaxis()
        ax.axis('off')
        fig.tight_layout(pad=0)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        return fig

    def display_colored_shadow_html(self):
        svg_parts = []
        
        view_box_width = self.n 
        view_box_height = self.num_rows
        if view_box_width == 0 or view_box_height == 0:
            return "<svg width='100%' height='100px'><text x='10' y='20'>Error: Diagram dimensions are zero.</text></svg>"

        line_stroke_width = 0.02
        colored_line_stroke_width = 0.05
        dot_radius = 0.08

        svg_parts.append(f'<svg viewBox="-0.5 -0.5 {view_box_width+1} {view_box_height+1}" xmlns="http://www.w3.org/2000/svg" style="width:100%; height:auto; background-color: white;">')

        if not self.seg_color and not self.color_list: # Ensure coloring if not done
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
        n = self.n
        # self.num_rows is always a multiple of n and self.num_rows >= n (actually >= 2n-1).
        # If num_rows == n, r0 = (1-1)*n = 0.
        # If num_rows == kn, r0 = (k-1)*n.
        r0 = (self.num_rows // n - 1) * n
        
        r1 = r0 + n

        Hw = [row[:] for row in self.H[r0:r1 + 1]]
        Vw = [row[:] for row in self.V[r0:r1]]
        dots = [(c, r - r0) for c, r in self.dots if r0 < r <= r1]
        
        segs, cols = [], {}
        for old_idx, seg in enumerate(self.segments):
            # Check if the segment is *within* the vertical bounds of the final window
            # A segment is in the window if both its y-endpoints are within [r0, r1]
            # or if it crosses the boundaries of the window.
            # For simplicity, we'll consider segments whose *midpoint* y-value is within the window,
            # or whose endpoints are within or cross the window boundaries.
            # A more precise way is to check if any part of the segment is in the window.
            # However, for boundary segments, we need their representation relative to the window.

            # We are interested in segments that are *part* of the final window's local grid.
            # The y-coordinates of segment endpoints are absolute.
            # We need to check if a segment's y-coordinates fall within the range [r0, r1].
            
            # A segment is part of the window if its y-coordinates are between r0 and r1.
            # For arcs, the center's y-coordinate matters.
            # For lines, both endpoints' y-coordinates matter.

            # Let's simplify: if any part of the segment is within the y-range [r0, r1]
            # and x-range [0, n], it's potentially part of the window.
            # The transformation (y-r0) will handle the relative positioning.

            # Filter segments that are fully contained or cross into the final window's y-range
            min_y_seg = min(ep[1] for ep in seg['endpoints'])
            max_y_seg = max(ep[1] for ep in seg['endpoints'])

            # A segment is considered part of the window if it's at least partially within its y-range
            # and fully within its x-range.
            # The x-coordinates of segments are already relative to the diagram's 0 to n width.
            if not (max_y_seg < r0 or min_y_seg > r1): # If segment y-range overlaps with window y-range
                new_seg_data = seg.copy()
                new_seg_data['endpoints'] = tuple((x, y - r0) for x, y in seg['endpoints'])
                
                if new_seg_data['type'] == 'line':
                    new_seg_data['points'] = new_seg_data['endpoints']
                else: # arc
                    cx, cy = new_seg_data['center']
                    new_seg_data['center'] = (cx, cy - r0)
                
                # Further filter: ensure the transformed segment is within the 0 to n bounds for x and y
                # This is important because a segment might cross the r0 or r1 boundary.
                # We only want the part of the segment that falls within the n x n window.
                # The current segment extraction logic is a bit too broad.
                # It should ideally clip segments to the window boundaries.
                # For now, we assume segments are either fully in or fully out based on their original definition
                # relative to the cells they belong to. The current filtering is based on original segment endpoints.

                # A segment is truly in the window if its original definition was based on cells
                # (c, r_cell) where r0 <= r_cell < r1.
                # The current segment generation iterates r from 0 to num_rows-1.
                # So, if a segment was generated from a cell (c, r_cell) where r0 <= r_cell < r1,
                # its y-coordinates will be around r_cell and r_cell+1.
                
                # Let's refine the condition: a segment belongs to the window if its original
                # "creation context" (the cell (c,r) in _compute_segments) was in the final window's rows.
                # This is implicitly handled if we consider the y-coordinates of its endpoints.
                # A segment is "in" the window if its y-coordinates are within [r0, r1]
                # and its x-coordinates are within [0, n].
                
                # The critical part is that `_get_boundary_color_sequence_from_data` looks for segments
                # at specific midpoints (e.g., (0, r_idx + 0.5)). So, the `segs` list for the window
                # must contain segments with these *local* coordinates.

                # Check if the *transformed* segment is within the 0 to n local grid
                local_min_y = min(ep[1] for ep in new_seg_data['endpoints'])
                local_max_y = max(ep[1] for ep in new_seg_data['endpoints'])
                local_min_x = min(ep[0] for ep in new_seg_data['endpoints'])
                local_max_x = max(ep[0] for ep in new_seg_data['endpoints'])

                # A segment is part of the window if it's at least partially within its local y-range [0, n]
                # and local x-range [0, n].
                # More strictly, for boundary checks, we need segments whose endpoints lie on the window boundary.
                if not (local_max_y < 0 or local_min_y > n or local_max_x < 0 or local_min_x > n):
                    idx_new = len(segs)
                    segs.append(new_seg_data)
                    if old_idx in self.seg_color:
                        cols[idx_new] = self.seg_color[old_idx]
                        
        return {'H':Hw, 'V':Vw, 'dots':dots, 'segments':segs, 'seg_color':cols}

    def _get_boundary_color_sequence_from_data(self, window_data, side):
        n_local = self.n 
        segs = window_data['segments']
        cols = window_data['seg_color'] 
        result = []

        def get_color_idx_from_window_cols(seg_idx_in_window):
            if seg_idx_in_window is None:
                return None
            color_str = cols.get(seg_idx_in_window)
            if color_str is None:
                return None
            if not self.color_list: 
                 return None
            try:
                return self.color_list.index(color_str) + 1
            except ValueError:
                return None 

        if side == 'left':
            for r_idx in range(n_local): 
                mid = (0, r_idx + 0.5)
                seg_idx = next((i for i, s in enumerate(segs) if any(abs(x_ep-mid[0])<1e-6 and abs(y_ep-mid[1])<1e-6 for x_ep,y_ep in s['endpoints'])), None)
                result.append(get_color_idx_from_window_cols(seg_idx))
        elif side == 'right':
            for r_idx in range(n_local):
                mid = (n_local, r_idx + 0.5)
                seg_idx = next((i for i, s in enumerate(segs) if any(abs(x_ep-mid[0])<1e-6 and abs(y_ep-mid[1])<1e-6 for x_ep,y_ep in s['endpoints'])), None)
                result.append(get_color_idx_from_window_cols(seg_idx))
        elif side == 'top':
            for c_idx in range(n_local):
                mid = (c_idx + 0.5, 0)
                seg_idx = next((i for i, s in enumerate(segs) if any(abs(x_ep-mid[0])<1e-6 and abs(y_ep-mid[1])<1e-6 for x_ep,y_ep in s['endpoints'])), None)
                result.append(get_color_idx_from_window_cols(seg_idx))
        elif side == 'bottom':
            for c_idx in range(n_local):
                mid = (c_idx + 0.5, n_local)
                seg_idx = next((i for i, s in enumerate(segs) if any(abs(x_ep-mid[0])<1e-6 and abs(y_ep-mid[1])<1e-6 for x_ep,y_ep in s['endpoints'])), None)
                result.append(get_color_idx_from_window_cols(seg_idx))
        return result

    def check_periodicity(self):
        """Checks if the final window is periodic in its coloring."""
        if self.num_rows < self.n: # Cannot check periodicity if diagram is smaller than one window
            # This case should ideally be prevented by affine_RS extending the diagram first.
            return False

        final_window_data = self.extract_final_window()
        
        # If C=0 (no colors), color_list is empty. _get_boundary_color_sequence_from_data will return Nones.
        # This means all boundaries will be [None, None, ...], which is periodic.
        left_colors = self._get_boundary_color_sequence_from_data(final_window_data, 'left')
        right_colors = self._get_boundary_color_sequence_from_data(final_window_data, 'right')
        top_colors = self._get_boundary_color_sequence_from_data(final_window_data, 'top')
        bottom_colors = self._get_boundary_color_sequence_from_data(final_window_data, 'bottom')
        
        return (left_colors == right_colors) and (top_colors == bottom_colors)

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
        # affine_RS will ensure coloring is done.
        final_window_data = self.extract_final_window()
        return self._get_boundary_color_sequence_from_data(final_window_data, 'left')

    def Q_tabloid_list(self):
        # affine_RS ensures coloring.
        final_window_data = self.extract_final_window()
        return self._get_boundary_color_sequence_from_data(final_window_data, 'bottom')

    def list_2_tabloid(self, lst):
        if lst is None: lst = []
        valid_entries = [v for v in lst if v is not None]
        maxl = max(valid_entries, default=0)
        return [[i+1 for i,v in enumerate(lst) if v==k] for k in range(1, maxl+1)]

    def P_tabloid(self):
        rows = self.list_2_tabloid(self.P_tabloid_list())
        return Tabloid(rows)

    def Q_tabloid(self):
        q_list = self.Q_tabloid_list()
        if q_list is None: q_list = [] 
        rows = self.list_2_tabloid(q_list[::-1])##
        return Tabloid(rows)

    def display_P_tabloid(self):
        print("P-tabloid:")
        self.P_tabloid().display()

    def display_Q_tabloid(self):
        print("Q-tabloid:")
        self.Q_tabloid().display()

    def lambda_list(self):
        # Lambda is read from the right boundary of the diagram *excluding* the final (periodic) window.
        # affine_RS ensures coloring is done.
        L_val_internal = [] 
        
        # Number of rows to consider for lambda is total rows minus one window height (n)
        rows_for_lambda = self.num_rows - self.n
        if rows_for_lambda < 0: # Should not happen if num_rows >= n
            rows_for_lambda = 0

        for row_idx_abs in range(rows_for_lambda): # Iterate over rows *before* the final window
            mid_abs = (self.n, row_idx_abs + 0.5) 
            
            idx = next((i for i, s_glob in enumerate(self.segments)
                        if any(abs(x_ep - mid_abs[0]) < 1e-6 and abs(y_ep - mid_abs[1]) < 1e-6 for x_ep, y_ep in s_glob['endpoints'])), None)
            
            if idx is None:
                L_val_internal.append(None)
            else:
                cstr = self.seg_color.get(idx)
                color_val = None
                if cstr and self.color_list: 
                    try:
                        color_val = self.color_list.index(cstr) + 1
                    except ValueError: 
                        color_val = None
                L_val_internal.append(color_val)
        
        valid_entries = [v for v in L_val_internal if v is not None]
        maxlab = max(valid_entries, default=0)
        return [L_val_internal.count(i) for i in range(1, maxlab + 1)]

    def affine_RS(self, display_output_console=True, max_extensions=10, debug_stop_after_extensions=None):
        # Ensure initial coloring for the first check_periodicity
        if not self.color_list and not self.seg_color:
            self.auto_color_left_sequence()

        extensions_done = 0
        for extension_count in range(max_extensions + 1): 
            # At this point, 'extension_count' extensions have been completed.
            # The diagram is in the state reflecting these completed extensions.
            extensions_done = extension_count

            if self.check_periodicity():
                break
            
            # Check for debug stop condition
            if debug_stop_after_extensions is not None and extension_count == debug_stop_after_extensions:
                if display_output_console:
                    print(f"DEBUG: Stopping after {extensions_done} extensions as requested. Current state is after these extensions.")
                break

            # If not periodic and not debug-stopped, check if max_extensions limit is reached for non-periodic diagram
            if extension_count == max_extensions:
                raise RuntimeError(f"Growth diagram did not become periodic within {max_extensions} extensions.")

            self.num_rows += self.n
            self.T = self.B + self.num_rows 

            # Recompute for the extended diagram
            self.compute_edge_labels() 
            self._compute_segments()   
            self._compute_components() 
            self._build_boundary_map() 
            
            # Recolor the entire extended diagram
            self.seg_color = {} 
            self.color_list = [] 
            self.auto_color_left_sequence()
        
        P = self.P_tabloid()
        Q = self.Q_tabloid()
        lam = self.lambda_list()

        # Check periodicity of the final state we stopped at
        final_state_is_periodic = self.check_periodicity()

        if display_output_console:
            print(f"--- Results after {extensions_done} extension(s) ---")
            if final_state_is_periodic:
                print(f"Periodic window found after {extensions_done} extension(s).")
            elif debug_stop_after_extensions is not None and extensions_done == debug_stop_after_extensions:
                print(f"Stopped after {extensions_done} extension(s) for debugging. Periodicity at this point: {final_state_is_periodic}.")
            # If RuntimeError was raised, this part isn't reached.
            self.display_P_tabloid()
            self.display_Q_tabloid()
            print(f"λ: {lam}")
            
        return P, Q, lam, extensions_done

def run_app(evt=None):
    res_elem = document.getElementById('result')
    diag_output_elem = document.getElementById('diagram-output-area')
    diag_output_elem.innerHTML = '' 
    res_elem.textContent = ''

    try:
        perm_input_value = document.getElementById('perm').value.strip()
        if not perm_input_value:
            res_elem.textContent = "Please enter an affine permutation."
            return

        L = [int(x) for x in perm_input_value.split(',') if x.strip()]
        w_display_string = f'w = [{perm_input_value}]\n\n'

        gd = GrowthDiagram(L)

        # Perform affine RS algorithm, which includes finding the periodic window
        P_tab, Q_tab, lam_list, num_extensions = gd.affine_RS(display_output_console=False, max_extensions=10)

        # Now 'gd' is in its final state, display it
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
            diag_output_elem.innerHTML = '<div id="colored-shadow-plot-target-single"></div>'
            svg_cs = gd.display_colored_shadow_html()
            document.getElementById('colored-shadow-plot-target-single').innerHTML = svg_cs

        out_P_rows = [' '.join(map(str, row)) for row in P_tab.rows] if P_tab.rows else ["(empty)"]
        out_Q_rows = [' '.join(map(str, row)) for row in Q_tab.rows] if Q_tab.rows else ["(empty)"]

        extension_info = f"Periodic window found after {num_extensions} extension(s).\n" if num_extensions >= 0 else ""

        output_text = w_display_string + extension_info + \
                      "P-tabloid:\n" + "\n".join(out_P_rows) + \
                      "\n\nQ-tabloid:\n" + "\n".join(out_Q_rows) + \
                      f"\n\nλ:\n{lam_list}"
        res_elem.textContent = output_text

    except ValueError as e:
        res_elem.textContent = f"Input Error: {e}"
    except RuntimeError as e: 
        res_elem.textContent = f"Processing Error: {e}"
    except Exception as e:
        res_elem.textContent = f"An unexpected error occurred: {e}\nCheck console for details."
        print(f"Full error in run_app: {type(e).__name__}: {e}")
        import traceback
        print(traceback.format_exc())

# --- PyScript specific setup ---
if __name__ == "__main__":
    # This part is for local testing if you run it as a standard Python script
    # It won't be executed in PyScript environment directly unless called.
    # For PyScript, the run_app function is typically triggered by an event.
    
    # Example usage for local testing (uncomment to use):
    # try:
    #     # Test case 1: Should become periodic quickly
    #     L_test1 = [2, 0, 1] 
    #     print(f"Testing with L = {L_test1}")
    #     gd1 = GrowthDiagram(L_test1)
    #     P1, Q1, lam1, ext1 = gd1.affine_RS(display_output_console=True, max_extensions=5)
    #     print(f"Extensions for L_test1: {ext1}\n")

    #     # Test case 2: Another example
    #     L_test2 = [3, 0, 1, 2]
    #     print(f"Testing with L = {L_test2}")
    #     gd2 = GrowthDiagram(L_test2)
    #     P2, Q2, lam2, ext2 = gd2.affine_RS(display_output_console=True, max_extensions=5)
    #     print(f"Extensions for L_test2: {ext2}\n")

    #     # Test case 3: Potentially requires more extensions or might not stabilize quickly
    #     # L_test3 = [4, 0, 5, 1, 6, 2, 7, 3] # n=8
    #     # print(f"Testing with L = {L_test3}")
    #     # gd3 = GrowthDiagram(L_test3)
    #     # P3, Q3, lam3, ext3 = gd3.affine_RS(display_output_console=True, max_extensions=10)
    #     # print(f"Extensions for L_test3: {ext3}\n")
        
    #     # Test case with no label 1 segments on left boundary initially
    #     L_test_no_color = [3,4,5] # n=3, all L[i] >= n
    #     print(f"Testing with L = {L_test_no_color} (no initial left boundary colors)")
    #     gd_nc = GrowthDiagram(L_test_no_color)
    #     P_nc, Q_nc, lam_nc, ext_nc = gd_nc.affine_RS(display_output_console=True, max_extensions=5)
    #     print(f"Extensions for L_test_no_color: {ext_nc}\n")


    # except Exception as e:
    #     print(f"Error during local test: {e}")
    #     import traceback
    #     print(traceback.format_exc())
    pass # PyScript handles the main execution flow via event bindings
