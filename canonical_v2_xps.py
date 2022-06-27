words = [{"text": "bonjour", "coords": [0,0,10,10]} for _ in range(300000)]

import time
start_time = time.time()

p = [w for i,w in enumerate(words) if 150_000<= i <150_995 ]


print("time elapsed: {:.2f}s".format(time.time() - start_time))
