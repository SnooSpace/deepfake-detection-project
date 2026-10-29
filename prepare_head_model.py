import struct, json, base64, numpy as np

with open('LeePerrySmith.glb', 'rb') as f:
    f.seek(12)
    json_len, _ = struct.unpack('<II', f.read(8))
    gltf = json.loads(f.read(json_len).decode('utf-8'))
    bin_len, _ = struct.unpack('<II', f.read(8))
    bin_data = f.read(bin_len)

prim = gltf['meshes'][0]['primitives'][0]
pos_acc = gltf['accessors'][prim['attributes']['POSITION']]
pos_bv = gltf['bufferViews'][pos_acc['bufferView']]
pos_offset = pos_bv.get('byteOffset', 0) + pos_acc.get('byteOffset', 0)
pos = np.frombuffer(bin_data[pos_offset : pos_offset + pos_acc['count'] * 12], dtype=np.float32).reshape(-1, 3).copy()

idx_acc = gltf['accessors'][prim['indices']]
idx_bv = gltf['bufferViews'][idx_acc['bufferView']]
idx_offset = idx_bv.get('byteOffset', 0) + idx_acc.get('byteOffset', 0)
faces = np.frombuffer(bin_data[idx_offset : idx_offset + idx_acc['count'] * 2], dtype=np.uint16).reshape(-1, 3).copy()

# Keep bust: y > -2.75
keep_mask = pos[:, 1] > -2.75
# Also remove outer shoulder flare: x > 3.2 or x < -3.2 when y < -1.2
shoulder_cut = (np.abs(pos[:, 0]) > 3.2) & (pos[:, 1] < -1.2)
keep_mask = keep_mask & (~shoulder_cut)

# Reindex vertices
new_indices_map = np.full(len(pos), -1, dtype=np.int32)
new_indices_map[keep_mask] = np.arange(np.sum(keep_mask))

valid_faces = []
for a, b, c in faces:
    if keep_mask[a] and keep_mask[b] and keep_mask[c]:
        valid_faces.append([new_indices_map[a], new_indices_map[b], new_indices_map[c]])

cropped_pos = pos[keep_mask].astype(np.float32)
cropped_faces = np.array(valid_faces, dtype=np.uint16)

print(f'Cropped vertices: {len(cropped_pos)}, cropped faces: {len(cropped_faces)}')

# Compute heat values for each vertex
def get_heat(p):
    x, y, z = p[0], p[1], p[2]
    # Forehead center
    df = ((x * 0.95)**2 + ((y - 2.45) * 1.05)**2 + ((z - 1.45) * 0.9)**2)**0.5
    hf = np.exp(-df**2 / (1.45**2))
    
    # Chin center
    dc = ((x * 1.1)**2 + ((y + 0.6) * 1.1)**2 + ((z - 1.7) * 0.95)**2)**0.5
    hc = np.exp(-dc**2 / (1.25**2))
    
    # Left cheek / jaw hot spot
    dlc = (((x + 0.7) * 1.2)**2 + ((y + 0.1) * 1.3)**2 + ((z - 1.6) * 1.0)**2)**0.5
    hlc = np.exp(-dlc**2 / (0.95**2)) * 0.85
    
    # Nose tip warm spot
    dn = ((x * 1.5)**2 + ((y - 0.75) * 1.8)**2 + ((z - 2.5) * 1.0)**2)**0.5
    hn = np.exp(-dn**2 / (0.7**2)) * 0.8
    
    frontness = max(0.0, min(1.0, (z + 0.3) / 1.5))
    h = max(hf * 1.15, hc * 1.1, hlc, hn) * frontness
    
    # Eye valley
    de = ((x * 1.4)**2 + ((y - 1.4) * 2.2)**2 + ((z - 2.0) * 1.0)**2)**0.5
    if de < 0.8:
        h *= (0.45 + 0.55 * (de / 0.8))
    return min(1.0, max(0.0, float(h)))

heat_arr = np.array([get_heat(p) for p in cropped_pos], dtype=np.float32)
heat_uint8 = (heat_arr * 255.0).round().astype(np.uint8)

# Center the model around chin/eyes
center = np.array([0.0, 0.2, 0.0], dtype=np.float32)
cropped_pos -= center

min_p = cropped_pos.min(axis=0)
max_p = cropped_pos.max(axis=0)
scale = max_p - min_p

norm_pos = ((cropped_pos - min_p) / scale * 65535.0).round().astype(np.uint16)

pos_bytes = norm_pos.tobytes()
idx_bytes = cropped_faces.flatten().tobytes()
heat_bytes = heat_uint8.tobytes()

b64_pos = base64.b64encode(pos_bytes).decode('ascii')
b64_idx = base64.b64encode(idx_bytes).decode('ascii')
b64_heat = base64.b64encode(heat_bytes).decode('ascii')

print(f'Total vertices: {len(cropped_pos)}, faces: {len(cropped_faces)}')
print(f'Base64 sizes: pos={len(b64_pos)/1024:.1f}KB, idx={len(b64_idx)/1024:.1f}KB, heat={len(b64_heat)/1024:.1f}KB')

js_content = f"""// Lee Perry Smith 3D Bust Mesh with Forensic Grad-CAM Attention Heatmap
window.HEAD_3D_DATA = {{
  numVerts: {len(cropped_pos)},
  numFaces: {len(cropped_faces)},
  min: [{min_p[0]:.6f}, {min_p[1]:.6f}, {min_p[2]:.6f}],
  max: [{max_p[0]:.6f}, {max_p[1]:.6f}, {max_p[2]:.6f}],
  posB64: "{b64_pos}",
  idxB64: "{b64_idx}",
  heatB64: "{b64_heat}"
}};
"""

with open('head_model.js', 'w', encoding='utf-8') as f:
    f.write(js_content)

print('head_model.js created successfully!')
