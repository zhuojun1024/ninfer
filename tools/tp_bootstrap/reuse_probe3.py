import json, time, urllib.request
URL = 'http://127.0.0.1:8088/v1/chat/completions'
S = 'You are a terse assistant.'
def post(msgs, mt=16):
    body = json.dumps({'model':'qwen3.8-27b','messages':msgs,'max_tokens':mt,'temperature':0,'stream':False}).encode()
    req = urllib.request.Request(URL, data=body, headers={'Content-Type':'application/json'})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    m = d['choices'][0]['message']
    print('  prompt=%d out=%d wall=%.2fs content=%r reasoning=%r' % (
        d['usage']['prompt_tokens'], d['usage']['completion_tokens'], time.time()-t0,
        (m.get('content') or '')[:60], (m.get('reasoning_content') or '')[:60]))
    return m.get('reasoning_content') or '', m.get('content') or ''

def turn(text, extra, tag):
    msgs = [{'role':'system','content':S},{'role':'user','content':text}] + extra
    print(tag)
    return post(msgs)

U1 = 'Name a color.'
print('turn1 (no cache expected)')
r1, c1 = post([{'role':'system','content':S},{'role':'user','content':U1}])
h = [{'role':'assistant','content':c1 or 'Blue'}]
h += [{'role':'user','content':'Name another color.'}]
print('turn2 (should reuse a boundary near U1 end)')
r2, c2 = post([{'role':'system','content':S},{'role':'user','content':U1}] + h)
h2 = h + [{'role':'assistant','content':c2 or 'Green'},{'role':'user','content':'Name a third color.'}]
print('turn3 (should reuse)')
r3, c3 = post([{'role':'system','content':S},{'role':'user','content':U1}] + h2)
turn3_fresh = None
print('invalidating request')
post([{'role':'system','content':S},{'role':'user','content':'What is 2+2?'}], 8)
print('turn3 again from scratch (no reuse)')
r3b, c3b = post([{'role':'system','content':S},{'role':'user','content':U1}] + h2)
print('MATCH' if (r3, c3) == (r3b, c3b) else 'MISMATCH %r/%r vs %r/%r' % ((r3,c3)+(r3b,c3b)))