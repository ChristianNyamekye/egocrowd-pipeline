import re
with open('/workspace/sim/render.py') as f:
    code = f.read()
code = code.replace('renderer.update_scene(data, camera="")', 'renderer.update_scene(data)')
with open('/workspace/sim/render.py', 'w') as f:
    f.write(code)
print('Fixed camera reference')
