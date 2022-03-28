import sys

input_path = './partial_lihkg.txt'
output_path = './partial_lihkg_no_dup.txt'

seen = set()
with open(input_path,encoding='utf-8',mode='r') as fin, open(output_path,mode='w',encoding='utf-8') as fout:
    while True:
        line = str(fin.readline())
        if len(line) == 0:
            break
        line_hash = hash(line)
        if line_hash not in seen:
            fout.write(line)
            seen.add(line_hash)

print(sys.getsizeof(seen))
