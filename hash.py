import hashlib
import json
import pathlib

chunks = ["test with this and check", "cache hit", "cache no hit"]
text = ''.join(chunks)
hash_umber = hashlib.sha256(text.encode()).hexdigest()
print(hash_umber)


cache_path = pathlib.Path("test.json")
if not cache_path.exists():
    print("not detected")
    with open("test.json", 'w') as f:
        json.dump({"hash_umber": hash_umber}, f)
else:
    with open("test.json", 'r') as f:
        dict = json.load(f)
    if dict['hash_umber'] == hash_umber:
        print("detected")
    else:
        print("not detected")
        with open("test.json", 'w') as f:
            json.dump({"hash_umber": hash_umber}, f)