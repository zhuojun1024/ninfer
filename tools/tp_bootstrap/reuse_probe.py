import json, urllib.request
URL = 'http://127.0.0.1:8088/v1/chat/completions'
S = 'You are a terse assistant.'
def post(msgs, mt=24):
    body = json.dumps({'model':'qwen3.8-27b','messages':msgs,'max_tokens':mt,'temperature':0,'stream':False}).encode()
    req = urllib.request.Request(URL, data=body, headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    m = d['choices'][0]['message']
    return m.get('reasoning_content') or '', m.get('content') or ''

hist = [{'role':'system','content':S}]
hist.append({'role':'user','content':'Name the first planet from the sun.'})
r1, c1 = post(hist)
print('turn1 reasoning=%r content=%r' % (r1[:80], c1))
# turn2 WITH reasoning echoed (preserve_thinking style)
h2 = hist + [{'role':'assistant','content':c1,'reasoning_content':r1}]
h2.append({'role':'user','content':'Now name the second planet.'})
r2, c2 = post(h2)
print('turn2 reasoning=%r content=%r' % (r2[:80], c2))
# turn3 WITH reasoning echoed
h3 = h2 + [{'role':'assistant','content':c2,'reasoning_content':r2}]
h3.append({'role':'user','content':'Now name the third planet.'})
r3, c3 = post(h3)
print('turn3 reasoning=%r content=%r' % (r3[:80], c3))
# turn4 WITHOUT reasoning (content only)
h4 = hist + [{'role':'assistant','content':c1}]
h4.append({'role':'user','content':'Now name the second planet.'})
r4, c4 = post(h4)
print('turn4(no-reasoning) reasoning=%r content=%r' % (r4[:80], c4))
# turn5 identical to turn4 again (to see lcp vs cached of turn4)
r5, c5 = post(h4)
print('turn5(no-reasoning repeat) content=%r' % (c5,))