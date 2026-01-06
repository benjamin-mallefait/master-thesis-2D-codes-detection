import sys
import yaml

if len(sys.argv) != 3:
    print("Usage: extract_yaml.py <dotted.key.path> <path_to_yaml>")
    sys.exit(1)

path = sys.argv[2]
key = sys.argv[1]

with open(path, "r") as f:
    data = yaml.safe_load(f)

for part in key.split('.'):
    data = data.get(part, None)
    if data is None:
        print("null")
        sys.exit(0)

print(data)