import json, urllib.request
URL = 'http://127.0.0.1:8088/v1/chat/completions'
def post(msgs, mt=8):
    body = json.dumps({'model':'qwen3.8-27b','messages':msgs,'max_tokens':mt,'temperature':0,'stream':False}).encode()
    req = urllib.request.Request(URL, data=body, headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())
S = 'You are a terse assistant.'
U = 'Name a color.'
m = [{'role':'system','content':S},{'role':'user','content':U}]
# identical request three times: LCP should equal the full previous prompt if rendering is deterministic
for i in range(3):
    d = post(m)
    print('same#%d prompt_tokens=%d' % (i, d['usage']['prompt_tokens']))
# then one with an extra assistant+user turn
m2 = m + [{'role':'assistant','content':'Blue'},{'role':'user','content':'Name another color.'}]
d = post(m2); print('extended prompt_tokens=%d' % d['usage']['prompt_tokens'])
# then the original again
d = post(m); print('same-again prompt_tokens=%d' % d['usage']['prompt_tokens'])