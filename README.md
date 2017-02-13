# gpgpu.js

```coffeescript
root = $("#gpgpu")
engine = getEngine root

kernel = "dst = src + tan(cos(sin(src * src)));"
input = [1, 2, 3, 4, 5]

output = engine.execute kernel, input
```
