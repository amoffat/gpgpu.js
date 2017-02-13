vertSrc = """
attribute vec2 pos;
varying vec2 texcoord;

void main() {
    gl_Position = vec4(pos, 0, 1);
    texcoord = (pos + vec2(1)) * vec2(0.5);
}
"""


getWebGL = (canvas) ->
    experimental = false
    gl = null

    try
        gl = canvas[0].getContext "webgl"
    catch x
        gl = null

    if gl == null
        try
            gl = canvas[0].getContext "experimental-webgl"
        catch x
            gl = null

    gl


# for debugging.  dumps out the currently bound textures to each texture unit
getCurrentTexBindings = (gl) ->
    map = {}
    maxTexUnits = gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS)
    
    for i in [0...maxTexUnits]
        tu = "TEXTURE#{i}"
        gl.activeTexture gl[tu]
        tex = gl.getParameter gl.TEXTURE_BINDING_2D
        map[i] = tex

    map


getGPUInfo = (gl) ->
    info =
        gpu: null
        vendor: null

    ext = getExt gl, "WEBGL_debug_renderer_info"
    if ext
        info.vendor = gl.getParameter(ext.UNMASKED_VENDOR_WEBGL)
        info.gpu = gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)

    info


createCanvas = (root, size, id) ->
    tag = "<canvas id='#{id}' width='#{size}' height='#{size}'></canvas>"
    el = $(tag)
    root.append el
    el

ensureCanvas = (root, size, id) ->
    canvas = $("##{id}")
    if not canvas.length
        canvas = createCanvas root, size, id
    canvas


# exceptions can be swallowed inside of promises.  logging also makes sure we
# can see the exception
raise = (msg) ->
    console.log "ERROR: #{msg}"
    throw msg

assert = (condition, message) ->
    if !condition
        raise message or "Assertion failed"

assertEqual = (a, b) ->
    assert _.isEqual(a, b), "#{a} not equal to #{b}"


# janky way of looking up the error we get back from gl.getError
lookupGLError = (gl, err) ->
    for key, val of gl
        if err == val
            return key

checkError = (gl) ->
    err = gl.getError()
    if err != gl.NO_ERROR
        glEnum = lookupGLError(gl, err)
        raise "GL error: #{glEnum}"

getUniformLocation = (gl, program, name) ->
    pos = gl.getUniformLocation program, name
    checkError gl
    assert pos != null, "#{name} not found"
    pos

getExt = (gl, name) ->
    gl.getExtension name

assertExt = (gl, name) ->
    ext = getExt gl, name
    assert ext, "#{name} extension required but not found."
    ext

createShader = (gl, src, type) ->
    shader = gl.createShader type
    checkError gl

    gl.shaderSource shader, src
    checkError gl

    gl.compileShader shader, src
    checkError gl

    status = gl.getShaderParameter shader, gl.COMPILE_STATUS
    checkError gl

    if not status
        log = gl.getShaderInfoLog shader
        checkError gl
        throw log

    shader


createProgram = (gl, vs, fs) ->
    prog = gl.createProgram()
    checkError gl

    gl.attachShader prog, vs
    checkError gl

    gl.attachShader prog, fs
    checkError gl

    gl.linkProgram prog
    checkError gl

    prog


createTexture = (gl, width, height) ->
    tex = gl.createTexture()
    checkError gl

    tex.width = width
    tex.height = height

    gl.bindTexture gl.TEXTURE_2D, tex
    checkError gl

    gl.texParameteri gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST
    checkError gl

    gl.texParameteri gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST
    checkError gl

    gl.texParameteri gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE
    checkError gl

    gl.texParameteri gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE
    checkError gl

    gl.texImage2D gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.FLOAT, null
    checkError gl

    gl.bindTexture gl.TEXTURE_2D, null

    tex

pixelTypeToArrayType = (gl, pixelType) ->
    m = {}
    m[gl.FLOAT] = Float32Array
    m[gl.UNSIGNED_BYTE] = Uint8Array

    m[pixelType]


createFBO = (gl, tex) ->
    fbo = gl.createFramebuffer()
    checkError gl

    gl.bindFramebuffer gl.FRAMEBUFFER, fbo
    checkError gl

    fbo.width = tex.width
    fbo.height = tex.height
    fbo.tex = tex
    
    gl.framebufferTexture2D gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0
    checkError gl

    assert gl.checkFramebufferStatus(gl.FRAMEBUFFER) == gl.FRAMEBUFFER_COMPLETE
    
    gl.bindFramebuffer gl.FRAMEBUFFER, null
    checkError gl

    fbo


determineTexSize = (numElements, maxSize) ->
    numPixels = Math.ceil numElements / 4
    numCols = Math.min maxSize, numPixels
    numRows = Math.ceil numPixels / maxSize

    [numCols, numRows]
    

getUniformSetter = (gl, type, val) ->
    name = "uniform#{type}"
    setter = gl[name]
    assert not _.isUndefined setter, name
    setter


getFBOType = (gl, fbo) ->
    gl.bindFramebuffer gl.FRAMEBUFFER, fbo
    checkError gl

    fboType = gl.getParameter(gl.IMPLEMENTATION_COLOR_READ_TYPE)

    gl.bindFramebuffer gl.FRAMEBUFFER, null
    checkError gl

    fboType

assertFloatFBO = (gl) ->
    tex = createTexture gl, 100, 100
    try
        fbo = createFBO gl, tex
    catch error
        throw """
Writing (using FBO) float textures unsupported.
If using Safari, try Chrome or Firefox on desktop.
"""

    type = getFBOType gl, fbo
    assert type == gl.FLOAT, """
Reading (readPixels) float textures unsupported.
If using Safari, try Chrome or Firefox on desktop.
"""



class @Storage

    # initializes storage from a html image element
    @fromImage: (engine, img, pixelType) ->
        gl = engine.gl
        pixelType = gl[pixelType]

        tex = createTexture gl, img.width, img.height

        gl.bindTexture gl.TEXTURE_2D, tex
        checkError gl
        gl.texImage2D gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, pixelType, img
        checkError gl

        gl.bindTexture gl.TEXTURE_2D, null

        storage = new Storage engine, tex, pixelType
        storage


    # initializes storage from an array
    @fromData: (engine, data) ->
        gl = engine.gl

        size = data.length
        storage = @fromSize engine, size

        storage.upload data
        storage


    # initializes storage from just an array size
    @fromSize: (engine, size, data=null) ->
        gl = engine.gl

        if (size instanceof Array)
            [width, height] = size
        else
            [width, height] = determineTexSize size, engine.maxSize

        tex = createTexture gl, width, height
        storage = new Storage engine, tex, gl.FLOAT

        if data
            storage.upload data

        storage


    constructor: (@engine, @tex, @dataType) ->
        @gl = @engine.gl
        @padding = 0
        @width = tex.width
        @height = tex.height

        # we need an FBO in order to do readPixels in download method
        @helperFBO = createFBO @gl, @tex


    upload: (array) ->
        if array.length % 4 != 0
            goodSize = Math.ceil(array.length / 4) * 4
            @padding = goodSize - array.length

            paddedArray = new Float32Array goodSize
            paddedArray.set array
            array = paddedArray

        maxSize = @engine.maxSize
        numPixels = array.length / 4
        if numPixels > maxSize and numPixels % maxSize != 0
            goodSize = Math.ceil(numPixels / maxSize) * maxSize * 4
            @padding = @padding + goodSize - array.length

            paddedArray = new Float32Array goodSize
            paddedArray.set array
            array = paddedArray

        @gl.bindTexture @gl.TEXTURE_2D, @tex
        checkError @gl

        @gl.texImage2D @gl.TEXTURE_2D, 0, @gl.RGBA, @width, @height, 0, @gl.RGBA, @gl.FLOAT, array
        checkError @gl

        @gl.bindTexture @gl.TEXTURE_2D, null


    download: (padding=@padding) ->
        @gl.bindFramebuffer @gl.FRAMEBUFFER, @helperFBO
        checkError @gl

        arrayType = pixelTypeToArrayType @gl, @dataType


        array = new arrayType @width*@height*4
        @gl.readPixels 0, 0, @width, @height, @gl.RGBA, @dataType, array
        checkError @gl

        @gl.bindFramebuffer @gl.FRAMEBUFFER, null
        checkError @gl

        if padding
            array = array.subarray 0, array.length-padding

        array



class @Computation
    constructor: (@engine, @size, @text, @preamble=null, @arity=1) ->
        @gl = @engine.gl

        if @arity == 1
            arityFullText = ""
            arityOneText = @text
        else
            arityFullText = @text
            arityOneText = ""

        if @preamble is null
            preambleText = ""
        else
            preambleText = @preamble

        fragSrc = """
        precision highp float;
        precision highp int;
        varying vec2 texcoord;

        uniform sampler2D tex_inputs;
        uniform ivec2 input_size;
        uniform int arity;

        #{preambleText}

        void execute(in float src, out float dst, in int execute_idx, in int offset_idx) {
            #{arityOneText}
        }

        void execute(in vec4 src, out vec4 dst, in int execute_idx, in int offset_idx) {
            #{arityFullText}
        }

        void main() {
            vec4 inputs = texture2D(tex_inputs, texcoord);
            vec4 outputs;
            int kernel_offset = int(gl_FragCoord.x) + int(gl_FragCoord.y) * input_size.x;

            if (arity == 1) {
                execute(inputs[0], outputs[0], kernel_offset, 0);
                execute(inputs[1], outputs[1], kernel_offset, 1);
                execute(inputs[2], outputs[2], kernel_offset, 2);
                execute(inputs[3], outputs[3], kernel_offset, 3);
            }
            // arity is 4 or "full"
            else {
                execute(inputs, outputs, kernel_offset, 0);
            }

            gl_FragColor = outputs;
        }
        """

        fragShader = createShader @gl, fragSrc, @gl.FRAGMENT_SHADER
        @program = createProgram @gl, @engine.vertShader, fragShader

    # performs the computation and download the results from GPU memory
    execute: (inputs, uniforms, output=null) ->
        output = @step inputs, uniforms, output
        results = output.download()
        results

    # performs the computation, but retains the output in GPU memory.  useful if
    # we're feeding the outputs as inputs to another computation
    step: (inputs, uniforms, output=null) ->
        assert (inputs instanceof Storage), "inputs are not Storage"


        if not output
            output = Storage.fromSize @engine, @size

        outFBO = createFBO @gl, output.tex

        # make sure our output's padding matches our input's padding, that way
        # when we download from the output, we only get the values that were
        # mapped 1-to-1 from the input
        output.padding = inputs.padding

        @gl.useProgram @program
        checkError @gl


        availTextureUnits = (i for i in [@engine.maxTexUnits-1..0])
        boundTextures = []

        getTexUnit = =>
            tu = availTextureUnits.pop()
            name = "TEXTURE#{tu}"
            [tu, @gl[name]]

        bindTexture = (tex, name) =>
            assert not _.isUndefined(tex), "tex for #{name} is undefined"

            [tu, tuEnum] = getTexUnit()
            @gl.activeTexture tuEnum
            checkError @gl

            @gl.bindTexture @gl.TEXTURE_2D, tex
            checkError @gl

            unbind = =>
                @gl.activeTexture tuEnum
                checkError @gl

                @gl.bindTexture @gl.TEXTURE_2D, null
                checkError @gl

            boundTextures.push unbind

            pos = getUniformLocation @gl, @program, name

            @gl.uniform1i pos, tu
            checkError @gl


        # bind our inputs to the program

        bindTexture inputs.tex, "tex_inputs"

        arityPos = getUniformLocation @gl, @program, "arity"
        @gl.uniform1i arityPos, @arity
        checkError @gl


        # we use the old getUniformLocation because we don't want to throw an
        # assertion if input_size isn't found.  it means it just isn't used in
        # the program
        pos = @gl.getUniformLocation @program, "input_size"
        checkError @gl
        if pos != null
            @gl.uniform2i pos, inputs.tex.width, inputs.tex.height
            checkError @gl

        @gl.viewport 0, 0, output.tex.width, output.tex.height
        checkError @gl


        # set our user-defined uniforms
        for name, [type, val] of uniforms
            pos = getUniformLocation @gl, @program, name

            if type == "s"
                tex = val.tex or val
                bindTexture tex, name
            else
                setter = getUniformSetter @gl, type, val
                setter.call @gl, pos, val
                checkError @gl


        @gl.bindFramebuffer @gl.FRAMEBUFFER, outFBO
        checkError @gl

        @engine.draw @program

        @gl.bindFramebuffer @gl.FRAMEBUFFER, null


        # maybe not necessary
        #for unbind in boundTextures
        #unbind()

        output




class Engine
    constructor: (@root) ->
        @canvas = ensureCanvas @root, 1, "gpgpu-canvas"
        @gl = getWebGL @canvas
        assert @gl != null, "WebGL not enabled."

        @gpuInfo = getGPUInfo @gl

        # let's determine the maximum column width of input data we can process
        # before we have to wrap around to a new row

        maxTexSize = @gl.getParameter @gl.MAX_TEXTURE_SIZE
        checkError @gl

        maxViewport = @gl.getParameter(@gl.MAX_VIEWPORT_DIMS)[0]
        checkError @gl

        @maxSize = Math.min maxTexSize, maxViewport

        @maxTexUnits = @gl.getParameter(@gl.MAX_TEXTURE_IMAGE_UNITS)


        assertExt @gl, "OES_texture_float"
        assertFloatFBO @gl


        # now let's set up our basic vertex shader and the plane geometry that
        # we'll pump through it on each execute

        @vertShader = createShader @gl, vertSrc, @gl.VERTEX_SHADER


        @buffer = @gl.createBuffer()
        checkError @gl

        @gl.bindBuffer @gl.ARRAY_BUFFER, @buffer
        checkError @gl

        plane = new Float32Array([-1, -1,  1, -1,  -1, 1,  -1, 1,  1, -1,  1, 1])
        @gl.bufferData @gl.ARRAY_BUFFER, plane, @gl.STATIC_DRAW

        @gl.bindBuffer @gl.ARRAY_BUFFER, null
        checkError @gl


    draw: (program) ->
        @gl.bindBuffer @gl.ARRAY_BUFFER, @buffer
        checkError @gl

        pos = @gl.getAttribLocation program, "pos"
        checkError @gl

        @gl.enableVertexAttribArray pos
        checkError @gl

        @gl.vertexAttribPointer pos, 2, @gl.FLOAT, false, 0, 0
        checkError @gl
        
        @gl.drawArrays @gl.TRIANGLES, 0, 6
        checkError @gl


    execute: (inputData, kernel, preamble=null, arity=1) ->
        inputs = Storage.fromData @, inputData
        comp = @createComputation inputData.length, kernel, preamble, arity

        comp.execute inputs


    createComputation: (size, kernel, preamble=null, arity=1) ->
        new Computation @, size, kernel, preamble, arity


@getEngine = (root) ->
    new Engine root

testBasic = (engine) ->
    inputs = new Float32Array [1, 2, 3, 4, 5]
    outputs = engine.execute inputs, "dst = src * src;"
    assertEqual outputs, new Float32Array [1, 4, 9, 16, 25]

testPingPong = (engine) ->
    inputData = new Float32Array [1, 2, 3, 4, 5]
    inputs = Storage.fromData engine, inputData

    comp = engine.createComputation inputData.length, "dst = src * src;"

    inputs = comp.step inputs
    inputs = comp.step inputs
    inputs = comp.step inputs

    outputs = inputs.download()

    assertEqual outputs, new Float32Array [1, 256, 6561, 65536, 390625]


testUploadArray = (engine) ->
    inputData = new Float32Array [1, 2, 3, 4, 5]
    inputs = Storage.fromData engine, inputData

    numsSize = 2
    preamble = """
uniform int nums[#{numsSize}];

int lookup(in int nums[#{numsSize}], in int idx) {
    for (int i=0; i<#{numsSize}; i++) {
        if (i==idx) {
            return nums[i];
        }
    }
}
"""

    kernel = """
dst = src * float(lookup(nums, execute_idx));
"""

    comp = engine.createComputation inputData.length, kernel, preamble
    outputs = comp.execute inputs, {nums:["1iv", new Int32Array [2, 3]]}
    assertEqual outputs, new Float32Array [2, 4, 6, 8, 15]


testUniforms = (engine) ->
    inputData = new Float32Array [1, 2, 3, 4, 5]
    inputs = Storage.fromData engine, inputData

    comp = engine.createComputation inputData.length, "dst = src * float(num);", "uniform int num;"
    outputs = comp.execute inputs, {num:["1i", 3]}
    assertEqual outputs, new Float32Array [3, 6, 9, 12, 15]

testDefinedFunction = (engine) ->
    inputs = new Float32Array [1, 2, 3, 4, 5]
    preamble = """
float cube(float v) {
    return v * v * v;
}
"""
    outputs = engine.execute inputs, "dst = cube(src);", preamble
    assertEqual outputs, new Float32Array [1, 8, 27, 64, 125]

testFullArity = (engine) ->
    inputs = new Float32Array [1, 2, 3, 4]
    outputs = engine.execute inputs, "dst[0] = dot(src, vec4(1.0));", "", 4
    assertEqual outputs, new Float32Array [10, 0, 0, 0]


testBitOn = (engine) ->
    preamble = """
uniform int check;

bool bit_on(int num, int bit_idx, out int bit_off) {
    int shift = int(pow(2.0, float(bit_idx)));
    int tmp = num / shift;
    bool bit_on = int(mod(float(tmp), 2.0)) != 0;

    bit_off = num;
    if (bit_on) {
        bit_off = num - int(pow(2.0, float(bit_idx)));
    }
    return bit_on;
}
"""
    inputData = new Float32Array [1<<0, 1<<1, 6]
    inputs = Storage.fromData engine, inputData

    kernel = """
int off = 0;
bool was_on = bit_on(int(src), check, off);
dst = float(off) + (was_on ? 1.0 : 0.0);
"""

    comp = engine.createComputation inputData.length, kernel, preamble
    outputs = comp.execute inputs, {check: ["1i", 1]}
    assertEqual outputs, new Float32Array [1, 1, 5]
    

    inputData = new Float32Array [1<<7, 1<<6, 1<<5]
    inputs = Storage.fromData engine, inputData

    outputs = comp.execute inputs, {check: ["1i", 7]}
    assertEqual outputs, new Float32Array [1, 64, 32]


    inputData = new Float32Array [1<<15, 1<<13, 1<<14]
    inputs = Storage.fromData engine, inputData
    outputs = comp.execute inputs, {check: ["1i", 14]}
    assertEqual outputs, new Float32Array [32768, 8192, 1]


testExtraTex = (engine) ->
    preamble = """
uniform sampler2D util;

float lookup(in vec4 vals, in int idx) {
    for (int i=0; i<4; i++) {
        if (i==idx) {
            return vals[i];
        }
    }
}
"""
    kernel = """
vec4 vals = texture2D(util, texcoord);
float val = lookup(vals, offset_idx);
dst = src * val;
"""
    inputData = new Float32Array [1, 2, 3, 4, 5]
    inputs = Storage.fromData engine, inputData

    utilData = new Float32Array [6, 7, 8, 20, 15]
    util = Storage.fromData engine, utilData

    comp = engine.createComputation inputData.length, kernel, preamble

    uniforms =
        "util": ["s", util]
    outputs = comp.execute inputs, uniforms
    assertEqual outputs, new Float32Array [6, 14, 24, 80, 75]


TEST_SUITE = [
    ["basic", testBasic]
    ["pingPong", testPingPong]
    ["uploadArray", testUploadArray]
    ["extraTex", testExtraTex]
    ["definedFunction", testDefinedFunction]
    ["fullArity", testFullArity]
    ["uniforms", testUniforms]
    ["bitOn", testBitOn]
]


benchmark = (engine, num, kernelCPU, kernelGPU) ->
    inputs = new Float32Array num
    for i in [0..(inputs.length-1)]
        inputs[i] = Math.random()

    # run CPU kernel

    resultsCPU = new Float32Array inputs.length
    start = performance.now()
    for num, i in inputs
        resultsCPU[i] = kernelCPU(num)
    elapsedCPU = performance.now()-start


    # run GPU kernel

    start = performance.now()
    inputData = Storage.fromData engine, inputs
    comp = engine.createComputation inputs.length, kernelGPU
    overheadGPUElapsed = performance.now() - start

    start = performance.now()
    results = comp.step inputData
    engine.gl.finish()
    end = performance.now()
    elapsedGPU = end-start

    start = performance.now()
    resultsGPU = results.download()
    overheadGPUElapsed += performance.now() - start

    # validate the results


    e = 0.00001
    for _, i in resultsCPU
        dataCorrect = Math.abs(resultsCPU[i] - resultsGPU[i]) < e
        if not dataCorrect
            break
    assert dataCorrect, "results did not match"
    assert resultsCPU.length == resultsGPU.length, \
        "result sizes differ #{resultsCPU.length} vs #{resultsGPU.length}"

    [elapsedCPU, elapsedGPU, overheadGPUElapsed]


meanAndVariance = (samples) ->
    samples = _.clone samples

    # remove the best and worst
    samples.sort()
    samples.pop()
    samples.shift()

    mean = _.mean samples
    [mean, 0]



benchmarkTrials = (trials, engine, num, kernelCPU, kernelGPU) ->
    elapsedCPUSamples = []
    elapsedGPUSamples = []
    elapsedGPUOverheadSamples = []

    for i in [1..trials]
        results = benchmark engine, num, kernelCPU, kernelGPU
        [elapsedCPU, elapsedGPU, overheadGPUElapsed] = results

        elapsedCPUSamples.push elapsedCPU
        elapsedGPUSamples.push elapsedGPU
        elapsedGPUOverheadSamples.push overheadGPUElapsed

    [meanCPU, varCPU] = meanAndVariance elapsedCPUSamples
    [meanGPU, varGPU] = meanAndVariance elapsedGPUSamples
    [meanGPUOverhead, varGPUOverhead] = meanAndVariance elapsedGPUOverheadSamples

    [[meanCPU, varCPU], [meanGPU, varGPU], [meanGPUOverhead, varGPUOverhead]]



runBenchmark = (samples) ->
    root = $("#gpgpu")
    engine = getEngine root

    #kernelCPU = (num) -> num * num
    #kernelGPU = "dst = src * src;"
    kernelCPU = (num) -> num + Math.tan(Math.cos(Math.sin(num * num)))
    kernelGPU = "dst = src + tan(cos(sin(src * src)));"

    [elapsedCPU, elapsedGPU, elapsedGPUOverhead] = benchmarkTrials 10, engine, samples, kernelCPU, kernelGPU

    gpuTotal = elapsedGPU[0] + elapsedGPUOverhead[0]
    speedup = elapsedCPU[0] / gpuTotal
    tSpeedup = elapsedCPU[0] / elapsedGPU[0]

    console.log "CPU:", elapsedCPU[0]
    console.log "GPU Total:", gpuTotal
    console.log "GPU Execution:", elapsedGPU[0]
    console.log "GPU IO:", elapsedGPUOverhead[0]
    console.log "Actual Speedup:", speedup
    console.log "Theoretical Speedup:", tSpeedup



runTests = ->
    root = $("#gpgpu")
    for [name, test] in TEST_SUITE
        console.log "running #{name}"
        engine = getEngine root
        test engine

#runTests()
#runBenchmark 100000000
