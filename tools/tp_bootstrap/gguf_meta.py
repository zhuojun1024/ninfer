import struct, sys
path = sys.argv[1]
with open(path,'rb') as f:
    data = f.read(1<<27)  # first 128MB for metadata + tensor info
magic = data[:4]
assert magic == b'GGUF', magic
version, = struct.unpack_from('<I', data, 4)
tensor_count, kv_count = struct.unpack_from('<QQ', data, 8)
off = 24
FMT = {0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
SZ = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
def read_str(off):
    ln, = struct.unpack_from('<Q', data, off); off += 8
    s = data[off:off+ln].decode('utf-8'); return s, off+ln
keys = []
for _ in range(kv_count):
    key, off = read_str(off)
    vt, = struct.unpack_from('<I', data, off); off += 4
    if vt == 8:
        val, off = read_str(off)
    elif vt == 9:
        et, = struct.unpack_from('<I', data, off); off += 4
        n, = struct.unpack_from('<Q', data, off); off += 8
        vals=[]
        for i in range(n):
            if et==8:
                v, off = read_str(off); vals.append(v)
            else:
                sz=SZ[et]; vals.append(struct.unpack_from('<'+FMT[et], data, off)[0]); off+=sz
        val = vals
    else:
        sz=SZ[vt]; val = struct.unpack_from('<'+FMT[vt], data, off)[0]; off+=sz
    keys.append((key,val))
print('version',version,'tensors',tensor_count,'kv',kv_count)
for k,v in keys:
    if any(t in k for t in ['head','n_embd','n_layer','n_head','n_ff','ssm','rope','vocab','conv','state','group','rank','dim','block','n_r','n_v','n_k','intermediate','experts','full_attn','recurrent','context','n_rot','layer_types']):
        print(' ',k,'=',v)
