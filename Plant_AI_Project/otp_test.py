import json, http.client
conn = http.client.HTTPConnection("localhost", 8000, timeout=10)
payload = json.dumps({"email": "john.doe@test.com"})
headers = {"Content-Type": "application/json"}
conn.request("POST", "/api/send-otp", payload, headers)
r = conn.getresponse()
print(r.status, r.reason)
print(r.read().decode())
