"""Script to produce an SVG diagram of a neural network."""

import argparse
from enum import IntFlag, auto
import json
import math
from typing import List, Mapping, NamedTuple, Optional, Union
import tempfile

import drawsvg

FONT_FAMILY = "arial"
MAX_NODES = 10
NODE_SIZE_SCALE = 0.7
SAMPLE_SPACING = 4
LABELS = True

BLOCK_STROKE = "#7f7f7f"
BLOCK_FILL = "#ebebeb"
LAYER_STROKE = "#7f7f7f"
LAYER_FILL = "#e1e1e1"
NODE_FILL = "#5b9bd5"
NODE_STROKE = "#41719c"
ACTION_FILL = "#5b9bd5"
TEXT_FILL = "#6868b6"
ARROW_FILL = "#ddd"
TRANSITION_FILL = "#fff"
TRANSITION_STROKE = "#818181"


def to_grayscale():
    global LAYER_STROKE, LAYER_FILL, NODE_FILL, NODE_STROKE, ACTION_FILL, TEXT_FILL, TRANSITION_STROKE
    LAYER_FILL = "#e1e1e1"
    LAYER_STROKE = "#000"
    NODE_FILL = "#999"
    NODE_STROKE = "#000"
    ACTION_FILL = "#000"
    TEXT_FILL = "#000"
    TRANSITION_STROKE = "#000"


Point = NamedTuple("Point", [("x", float), ("y", float)])
Size = NamedTuple("Size", [("width", float), ("height", float)])


class Vector2:
    def __init__(self, x1: float, y1: float, x2: float, y2: float) -> None:
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.dx = x2 - x1
        self.dy = y2 - y1
        self.length = math.sqrt(self.dx * self.dx + self.dy * self.dy)
        self.dx /= self.length
        self.dy /= self.length

    def perp(self) -> 'Vector2':
        if self.dx != 0:
            return Vector2(0, 0, self.dy, -self.dx)
        else:
            return Vector2(0, 0, -self.dy, self.dx)

    def getPoint(self, distance: float) -> Point:
        return Point(self.x1 + distance * self.length * self.dx,
                     self.y1 + distance * self.length * self.dy)


def arrow(dir: Vector2, height: float, vertical=False) -> str:
    x1, y1 = dir.getPoint(0)
    x2, y2 = dir.getPoint(1)
    if vertical:
        space = 0.5 * height * dir.dx / dir.dy + height
        if dir.dy > 0:
            y1 += space / 2
            y2 -= space / 4
        else:
            y1 -= space / 2
            y2 += space / 4
    else:
        space = 0.5 * height * abs(dir.dy / dir.dx) + height
        if dir.dx > 0:
            x1 += space / 2
            x2 -= space / 4
        else:
            x1 -= space / 2
            x2 += space / 4

    dir = Vector2(x1, y1, x2, y2)

    perp = dir.perp()

    arrowHeight = 0.9 * height
    barHeight = 0.5 * height
    barLength = dir.length - height

    x = x1
    y = y1

    x += barHeight * perp.dx
    y += barHeight * perp.dy
    path = "M" + str(x) + "," + str(y)
    x += barLength * dir.dx
    y += barLength * dir.dy
    path += "L" + str(x) + "," + str(y)
    x += (arrowHeight - barHeight) * perp.dx
    y += (arrowHeight - barHeight) * perp.dy
    path += "L" + str(x) + "," + str(y)
    path += "L" + str(x2) + "," + str(y2)
    x -= (arrowHeight * 2) * perp.dx
    y -= (arrowHeight * 2) * perp.dy
    path += "L" + str(x) + "," + str(y)
    x += (arrowHeight - barHeight) * perp.dx
    y += (arrowHeight - barHeight) * perp.dy
    path += "L" + str(x) + "," + str(y)
    x -= barLength * dir.dx
    y -= barLength * dir.dy
    path += "L" + str(x) + "," + str(y)
    path += "Z"

    return path


def attachTop(middle: float, top: float, text: str, font_size: float) -> drawsvg.Text:
    element = drawsvg.Text(text, font_size, middle, top -
                           0.7 * 6, fill=TEXT_FILL, text_anchor="middle", font_family=FONT_FAMILY)
    return element


def attachBottom(middle: float, bottom: float, text: str, font_size: float) -> drawsvg.Text:
    element = drawsvg.Text(text, font_size, middle, bottom +
                           font_size, fill=TEXT_FILL, text_anchor="middle", font_family=FONT_FAMILY)

    return element


class Transition(IntFlag):
    Linear = auto()
    Sigmoid = auto()
    TanH = auto()
    LeakyReLU = auto()
    ReLU = auto()
    SiLU = auto()
    GELU = auto()
    Identity = auto()
    Dropout = auto()
    Token = auto()
    Previous = auto()

    @staticmethod
    def create(info: Mapping) -> "Transition":
        name = info.get("transition", None)

        if name is None:
            return Transition.Linear

        if "Sigmoid" in name:
            t = Transition.Sigmoid
        if "TanH" in name:
            t = Transition.TanH
        elif "LeakyReLU" in name:
            t = Transition.LeakyReLU
        elif "ReLU" in name:
            t = Transition.ReLU
        elif "SiLU" in name:
            t = Transition.SiLU
        elif "GELU" in name:
            t = Transition.GELU
        elif "Identity" in name:
            t = Transition.Identity
        elif "Token" in name:
            t = Transition.Token
        elif "Previous" in name:
            t = Transition.Previous
        else:
            t = Transition.Linear

        if "Dropout" in name:
            t |= Transition.Dropout

        return t


def layoutTransition(d: drawsvg.Drawing, transition: Transition, x1: float, y1: float,
                     x2: float, y2: float, height: float, vertical=False) -> None:
    dir = Vector2(x1, y1, x2, y2)

    d.append(drawsvg.Path(arrow(dir, height, vertical), fill=ARROW_FILL, stroke="none"))

    if transition == Transition.Linear or transition == Transition.Identity:
        return

    center = dir.getPoint(0.5)
    cx = center[0]
    cy = center[1]
    radius = height

    if transition & Transition.Dropout:
        stroke_dasharray = f"{radius * 0.5} {radius * 0.4}"
    else:
        stroke_dasharray = None

    d.append(drawsvg.Circle(cx, cy, radius,
                            fill=TRANSITION_FILL, stroke=TRANSITION_STROKE,
                            stroke_dasharray=stroke_dasharray))

    if transition & Transition.Sigmoid:
        layoutSigmoid(d, dir, height)
    if transition & Transition.TanH:
        layoutTanH(d, dir, height)
    elif transition & Transition.LeakyReLU:
        layoutLeakyReLU(d, dir, height)
    elif transition & Transition.ReLU:
        layoutReLU(d, dir, height)
    elif transition & Transition.GELU:
        layoutGELU(d, dir, height)
    elif transition & Transition.SiLU:
        layoutSiLU(d, dir, height)
    elif transition & Transition.Token:
        layoutToken(d, dir, height)
    elif transition & Transition.Previous:
        layoutPrevious(d, dir, height)


def layoutSigmoid(d: drawsvg.Drawing, direction: Vector2, height: float) -> None:
    ANGLE = math.pi * 0.25

    center = direction.getPoint(0.5)
    cx = center[0]
    cy = center[1]
    radius = height

    pathX = math.cos(ANGLE) * radius * 0.8
    pathY = math.sin(ANGLE) * radius * 0.8
    path = "".join([
        "M",
        str(cx - pathX),
        ",",
        str(cy),
        "C",
        str(cx + pathX),
        ",",
        str(cy),
        ",",
        str(cx - pathX),
        ",",
        str(cy - pathY),
        ",",
        str(cx + pathX),
        ",",
        str(cy - pathY)])
    d.append(drawsvg.Path(path, stroke="#000", strokewidth=1, fill="none"))


def layoutTanH(d: drawsvg.Drawing, direction: Vector2, height: float) -> None:
    ANGLE = math.pi * 0.25

    center = direction.getPoint(0.5)
    cx = center[0]
    cy = center[1]
    radius = height

    pathX = math.cos(ANGLE) * radius * 0.8
    pathY = math.sin(ANGLE) * radius * 0.8
    path = "".join([
        "M",
        str(cx - pathX),
        ",",
        str(cy + pathY),
        "C",
        str(cx + pathX),
        ",",
        str(cy + pathY),
        ",",
        str(cx - pathX),
        ",",
        str(cy - pathY),
        ",",
        str(cx + pathX),
        ",",
        str(cy - pathY)])
    d.append(drawsvg.Path(path, stroke="#000", strokewidth=1, fill="none"))


def layoutSiLU(d: drawsvg.Drawing, direction: Vector2, height: float) -> None:
    OUT_ANGLE = math.pi * 0.25

    center = direction.getPoint(0.5)
    cx = center[0]
    cy = center[1]

    radius = height

    leftX = cx - radius * 0.7
    curveX = cx - radius * 0.3
    bottomX = cx - math.cos(OUT_ANGLE) * radius * 0.15
    bottomY = cy + math.sin(OUT_ANGLE) * radius * 0.15
    outX = cx + math.cos(OUT_ANGLE) * radius * 0.7
    outY = cy - math.sin(OUT_ANGLE) * radius * 0.7
    path = "".join([
        "M",
        str(leftX), ",", str(cy),
        "H", str(curveX),
        "Q", str(bottomX), ",", str(bottomY), ",", str(cx), ",", str(cy),
        "L", str(outX), ",", str(outY)])
    d.append(drawsvg.Path(path, stroke="#000", strokewidth=1, fill="none"))


def layoutGELU(d: drawsvg.Drawing, direction: Vector2, height: float) -> None:
    OUT_ANGLE = math.pi * 0.25

    center = direction.getPoint(0.5)
    cx = center[0]
    cy = center[1]

    radius = height

    leftX = cx - radius * 0.7
    curveX = cx - radius * 0.3
    bottomX = cx - math.cos(OUT_ANGLE) * radius * 0.15
    bottomY = cy + math.sin(OUT_ANGLE) * radius * 0.15
    outX = cx + math.cos(OUT_ANGLE) * radius * 0.7
    outY = cy - math.sin(OUT_ANGLE) * radius * 0.7
    path = "".join([
        "M",
        str(leftX), ",", str(cy),
        "H", str(curveX),
        "Q", str(bottomX), ",", str(bottomY), ",", str(cx), ",", str(cy),
        "L", str(outX), ",", str(outY)])
    d.append(drawsvg.Path(path, stroke="#000", strokewidth=1, fill="none"))


def layoutLeakyReLU(d: drawsvg.Drawing, direction: Vector2, height: float) -> None:
    OUT_ANGLE = math.pi * 0.25
    IN_ANGLE = OUT_ANGLE * 0.25

    center = direction.getPoint(0.5)
    cx = center[0]
    cy = center[1]
    radius = height

    inpathX = math.cos(IN_ANGLE) * radius * 0.7
    inpathY = math.sin(IN_ANGLE) * radius * 0.7
    outpathX = math.cos(OUT_ANGLE) * radius * 0.7
    outpathY = math.sin(OUT_ANGLE) * radius * 0.7
    path = "".join([
        "M",
        str(cx - inpathX),
        ",",
        str(cy + inpathY),
        "L",
        str(cx), ",", str(cy),
        "L",
        str(cx + outpathX),
        ",",
        str(cy - outpathY)])
    d.append(drawsvg.Path(path, stroke="#000", strokewidth=1, fill="none"))


def layoutReLU(d: drawsvg.Drawing, direction: Vector2, height: float) -> None:
    ANGLE = math.pi * 0.25

    center = direction.getPoint(0.5)
    cx = center[0]
    cy = center[1]
    radius = height

    pathX = math.cos(ANGLE) * radius * 0.7
    pathY = math.sin(ANGLE) * radius * 0.7
    path = "".join([
        "M",
        str(cx - radius * 0.7),
        ",",
        str(cy),
        "H",
        str(cx),
        "L",
        str(cx + pathX),
        ",",
        str(cy - pathY)])
    d.append(drawsvg.Path(path, stroke="#000", strokewidth=1, fill="none"))


def layoutToken(d: drawsvg.Drawing, direction: Vector2, height: float) -> None:
    shift = height * 0.5 / direction.length
    points = [direction.getPoint(0.5 + i * shift) for i in range(-1, 2)]

    for p in points:
        d.append(drawsvg.Circle(p[0], p[1], height, fill=TRANSITION_FILL, stroke=TRANSITION_STROKE))

    cx, cy = points[-1]
    fontSize = height
    d.append(drawsvg.Text("[t]", fontSize, cx, cy + 0.25 * height,
                          fill="#000", text_anchor="middle", font_family='monospace'))


def layoutPrevious(d: drawsvg.Drawing, direction: Vector2, height: float) -> None:
    center = direction.getPoint(0.5)
    cx = center[0]
    cy = center[1]

    fontSize = height
    d.append(drawsvg.Text("t-1", fontSize, cx, cy + 0.25 * height,
                          fill="#000", text_anchor="middle", font_family='monospace'))


class Layer:
    def __init__(self, output: int, transition: Transition, unit_count=1):
        self.output = output
        self.transition = transition
        self.unit_count = unit_count
        self.inputs = []
        self.outputs = []

    def measure(self, unit_width: float, network_height: float, max_height: float,
                max_outputs: int, allow_full: bool) -> Size:
        height = math.log(self.output) * network_height / math.log(max_outputs)
        return Size(unit_width * self.unit_count, height)

    def addInput(self, x: Union[Point, float], y: Optional[float] = None) -> None:
        if isinstance(x, Point):
            self.inputs.append(x)
        elif y is not None:
            self.inputs.append(Point(x, y))

    def addOutput(self, x: Union[Point, float], y: Optional[float] = None) -> None:
        if isinstance(x, Point):
            self.outputs.append(x)
        elif y is not None:
            self.outputs.append(Point(x, y))

    def layout(self, x: float, y: float, width: float, height: float,
               index: int, d: drawsvg.Drawing, allow_full: bool) -> None:
        self.addInput(x, y + 0.5 * height)
        self.addOutput(x + width, y + 0.5 * height)


def layout_text(d: drawsvg.Drawing, text: str, x: float, y: float,
                width: float, height: float,
                fill=None, stroke=None,
                text_fill=None, mask=None,
                dashed=False) -> None:
    group = drawsvg.Group(mask=mask)
    node_size = width * NODE_SIZE_SCALE
    fill = fill or LAYER_FILL
    stroke = stroke or LAYER_STROKE
    text_fill = text_fill or ACTION_FILL
    stroke_dasharray = f"{node_size * 0.5} {node_size * 0.4}" if dashed else None
    group.append(drawsvg.Rectangle(x, y, width, height,
                                   fill=fill, stroke=stroke,
                                   stroke_dasharray=stroke_dasharray))
    font_size = node_size * 0.8
    cx = x + 0.5 * width
    cy = y + 0.5 * height + 0.3 * font_size
    transform = ""
    if len(text) > 1:
        cx += 0.3 * font_size
        cy -= 0.3 * font_size
        transform = f"rotate(-90, {cx}, {cy})"

    group.append(drawsvg.Text(text, font_size, cx, cy,
                                transform=transform, fill=text_fill, text_anchor="middle", font_family=FONT_FAMILY))
    d.append(group)


def layout_linear(d: drawsvg.Drawing, x: float, y: float, width: float, height: float, maxNodeCount: int) -> None:
    nodeSize = width * NODE_SIZE_SCALE
    nodeRadius = 0.35 * nodeSize
    nodeCount = min(math.floor(height / nodeSize), maxNodeCount)
    whiteSpace = height - (nodeCount * 2 * nodeRadius)
    whiteSpace /= nodeCount + 1
    d.append(drawsvg.Rectangle(x, y, width, height,
             rx=4, ry=4, fill=LAYER_FILL, stroke=LAYER_STROKE))

    yy = y + whiteSpace + nodeRadius
    xx = x + 0.5 * width
    for _ in range(nodeCount):
        d.append(drawsvg.Circle(xx, yy, nodeRadius,
                 fill=NODE_FILL, stroke=NODE_STROKE))
        yy += 2 * nodeRadius + whiteSpace


class Input(Layer):
    def __init__(self, name: str, nodeCount: int, transition: Transition) -> None:
        Layer.__init__(self, nodeCount, transition)
        self.name = name
        self.nodeCount = nodeCount

    def measure(self, unit_width: float, network_height: float, max_height: float,
                max_outputs: int, allow_full: bool) -> Size:
        node_size = unit_width * NODE_SIZE_SCALE
        width, height = Layer.measure(self, unit_width, network_height, max_height, max_outputs, allow_full)
        if height > node_size * self.output:
            height = max(node_size * self.output, 60)

        return Size(width, max(height, 30))

    def layout(self, x: float, y: float, width: float, height: float,
               index: int, d: drawsvg.Drawing, allow_full: bool) -> None:
        Layer.layout(self, x, y, width, height, index, d, allow_full)
        layout_text(d, self.name, x, y, width, height, dashed=True)

        if LABELS:
            font_size = width * NODE_SIZE_SCALE * 0.6
            d.append(attachBottom(x + 0.5 * width, y + height, str(self.nodeCount), font_size))


class Input2D(Layer):
    def __init__(self, name: str, channels: int, width: int, height: int,
                 transition: Transition) -> None:
        Layer.__init__(self, channels * width * height, transition)
        self.name = name
        self.channels = channels
        self.width = width
        self.height = height

    def layout(self, x: float, y: float, width: float, height: float,
               index: int, d: drawsvg.Drawing, allow_full: bool) -> None:
        Layer.layout(self, x, y, width, height, index, d, allow_full)
        layout_text(d, self.name, x, y, width, height, dashed=True)

        if LABELS:
            font_size = width * NODE_SIZE_SCALE * 0.6
            d.append(attachBottom(x + 0.5 * width, y + height,
                                "{}x{}x{}".format(self.channels, self.width, self.height), font_size))


class FullyConnected(Layer):
    def __init__(self, nodeCount: int, transition: Transition) -> None:
        Layer.__init__(self, nodeCount, transition)

    def measure(self, unit_width: float, network_height: float, max_height: float,
                max_outputs: int, allow_full: bool) -> Size:
        node_size = unit_width * NODE_SIZE_SCALE
        width, height = Layer.measure(self, unit_width, network_height, max_height, max_outputs, allow_full)
        display_count = math.floor(height / node_size)
        if display_count > self.output:
            display_count = self.output
            height = display_count * node_size
        elif display_count < self.output and self.output <= MAX_NODES and allow_full:
            display_count = self.output
            height = self.output * node_size

        if height > node_size * self.output:
            height = max(node_size * self.output, 60)

        return Size(width, max(height, 30))

    def layout(self, x: float, y: float, width: float, height: float,
               index: int, d: drawsvg.Drawing, allow_full: bool) -> None:
        Layer.layout(self, x, y, width, height, index, d, allow_full)
        layout_linear(d, x, y, width, height, self.output)

        if LABELS:
            font_size = width * NODE_SIZE_SCALE * 0.6
            d.append(attachTop(x + 0.5 * width, y, "F{}[{}]".format(index, self.output), font_size))
            d.append(attachBottom(x + 0.5 * width, y + height, str(self.output), font_size))


class Convolutional(Layer):
    def __init__(self, nodeCount: int, width: int, height: int, size: int,
                 stride: int, padding: int, transition: Transition) -> None:
        Layer.__init__(self, nodeCount, transition)
        self.width = width
        self.height = height
        self.size = size
        self.stride = stride
        self.padding = padding
        self.outputWidth = math.floor(
            (width + 2 * padding - size) / stride) + 1
        self.outputHeight = math.floor(
            (height + 2 * padding - size) / stride) + 1
        self.output = self.outputWidth * self.outputHeight * nodeCount
        self.nodeCount = nodeCount

    def measure(self, unit_width: float, network_height: float, max_height: float,
                max_outputs: int, allow_full: bool) -> Size:
        node_size = unit_width * NODE_SIZE_SCALE
        width, height = Layer.measure(self, unit_width, network_height, max_height, max_outputs, allow_full)
        display_count = math.floor(height / node_size)
        if display_count > self.output:
            display_count = self.output
            height = display_count * node_size
        elif display_count < self.output and self.output <= MAX_NODES and allow_full:
            display_count = self.output
            height = self.output * node_size

        return Size(width, height)

    def layout(self, x: float, y: float, width: float, height: float,
               index: int, d: drawsvg.Drawing, allow_full: bool) -> None:
        Layer.layout(self, x, y, width, height, index, d, allow_full)
        d.append(drawsvg.Rectangle(x, y, width, height,
                 rx=4, ry=4, fill=LAYER_FILL, stroke=LAYER_STROKE))

        nodeSize = width * NODE_SIZE_SCALE
        nodeWidth = 0.7 * nodeSize
        displayCount = math.floor(height / nodeSize)

        whiteSpace = height - (displayCount * nodeWidth)
        whiteSpace /= displayCount + 1

        yy = y + whiteSpace
        xx = x + 0.5 * (width - nodeWidth)

        for _ in range(displayCount):
            d.append(drawsvg.Rectangle(xx, yy, nodeWidth,
                     nodeWidth, fill=NODE_FILL, stroke=NODE_STROKE))
            yy += nodeWidth + whiteSpace

        if LABELS:
            font_size = nodeSize * 0.6
            d.append(attachTop(x + 0.5 * width, y, "C{}[{}x{}{}]".format(
                index, self.size, self.size,
                "@{}".format(self.stride) if self.stride > 1 else ""), font_size))
            d.append(attachBottom(x + 0.5 * width, y + height,
                                "{}x{}x{}".format(self.nodeCount, self.outputWidth,
                                                    self.outputHeight), font_size))


class ConvolutionalTranspose(Layer):
    def __init__(self, nodeCount: int, width: int, height: int, size: int,
                 stride: int, padding: int, output_padding: int, transition: Transition) -> None:
        Layer.__init__(self, nodeCount, transition)
        self.width = width
        self.height = height
        self.size = size
        self.stride = stride
        self.padding = padding
        self.outputWidth = stride * width - padding + output_padding
        self.outputHeight = stride * height - padding + output_padding
        self.output = self.outputWidth * self.outputHeight * nodeCount
        self.nodeCount = nodeCount

    def measure(self, unit_width: float, network_height: float, max_height: float,
                max_outputs: int, allow_full: bool) -> Size:
        node_size = unit_width * NODE_SIZE_SCALE
        width, height = Layer.measure(self, unit_width, network_height, max_height, max_outputs, allow_full)
        display_count = math.floor(height / node_size)
        if display_count > self.output:
            display_count = self.output
            height = display_count * node_size
        elif display_count < self.output and self.output <= MAX_NODES and allow_full:
            display_count = self.output
            height = self.output * node_size

        return Size(width, height)

    def layout(self, x: float, y: float, width: float, height: float,
               index: int, d: drawsvg.Drawing, allow_full: bool) -> None:
        Layer.layout(self, x, y, width, height, index, d, allow_full)
        d.append(drawsvg.Rectangle(x, y, width, height,
                 rx=4, ry=4, fill=LAYER_FILL, stroke=LAYER_STROKE))

        nodeSize = width * NODE_SIZE_SCALE
        nodeWidth = 0.7 * nodeSize
        displayCount = math.floor(height / nodeSize)

        whiteSpace = height - (displayCount * nodeWidth)
        whiteSpace /= displayCount + 1

        yy = y + whiteSpace
        xx = x + 0.5 * (width - nodeWidth)

        for _ in range(displayCount):
            d.append(drawsvg.Rectangle(xx, yy, nodeWidth,
                     nodeWidth, fill=NODE_FILL, stroke=NODE_STROKE))
            yy += nodeWidth + whiteSpace

        if LABELS:
            font_size = nodeSize * 0.6
            d.append(attachTop(x + 0.5 * width, y, "T{}[{}x{}{}]".format(
                index, self.size, self.size,
                "@{}".format(self.stride) if self.stride > 1 else ""), font_size))
            d.append(attachBottom(x + 0.5 * width, y + height,
                                "{}x{}x{}".format(self.nodeCount, self.outputWidth,
                                                    self.outputHeight), font_size))


class Pooling(Layer):
    def __init__(self, op: str, channels: int, width: int, height: int,
                 size: int, stride: int, padding: int,
                 transition: Transition) -> None:
        Layer.__init__(self, channels * math.floor((width + 2 * padding - size) / stride + 1)
                       * math.floor((height + 2 * padding - size) / stride + 1), transition)
        self.op = op
        self.channels = channels
        self.width = width
        self.height = height
        self.size = size
        self.stride = stride
        self.padding = padding
        self.outputWidth = math.floor(
            (width + 2 * padding - size) / stride) + 1
        self.outputHeight = math.floor(
            (height + 2 * padding - size) / stride) + 1

    def layout(self, x: float, y: float, width: float, height: float, index: int,
               d: drawsvg.Drawing, allow_full: bool) -> None:
        Layer.layout(self, x, y, width, height, index, d, allow_full)
        layout_text(d, self.op, x, y, width, height)
        font_size = width * NODE_SIZE_SCALE * 0.6
        if LABELS:
            d.append(attachTop(x + 0.5 * width, y, "P{}[{}x{}{}]".format(
                index, self.size, self.size, "@{}".format(self.stride) if self.stride > 1 else ""), font_size))
            d.append(attachBottom(x + 0.5 * width, y + height,
                                "{}x{}x{}".format(self.channels, self.outputWidth, self.outputHeight), font_size))


class OneHot(Layer):
    def __init__(self, embedSize: int, transition: Transition) -> None:
        Layer.__init__(self, embedSize, transition)
        self.embedSize = embedSize

    def layout(self, x: float, y: float, width: float, height: float, index: int,
               d: drawsvg.Drawing, allow_full: bool) -> None:
        Layer.layout(self, x, y, width, height, index, d, allow_full)
        layout_text(d, "embed", x, y, width, height)
        font_size = width * NODE_SIZE_SCALE * 0.6
        d.append(attachBottom(x + 0.5 * width, y + height,
                              "{}".format(self.embedSize), font_size))


class ResBlock(Layer):
    def __init__(self, nodeCount: int, width: int, height: int, stride: int,
                 transition: Transition) -> None:
        Layer.__init__(self, 0, transition, 6)
        self.nodeCount = nodeCount
        self.width = width
        self.height = height
        self.stride = stride
        self.outputWidth = width // stride
        self.outputHeight = height // stride
        self.output = self.outputWidth * self.outputHeight * nodeCount

    def layout(self, x: float, y: float, width: float, height: float, index: int,
               d: drawsvg.Drawing, allow_full: bool) -> None:
        unit = width / self.unit_count

        nodeSize = unit * NODE_SIZE_SCALE
        subHeight = 0.5 * height - 0.25 * unit
        nodeWidth = 0.7 * nodeSize
        displayCount = math.floor(subHeight / nodeSize)

        whiteSpace = subHeight - (displayCount * nodeWidth)
        whiteSpace /= displayCount + 1

        # add top
        # register input
        d.append(drawsvg.Rectangle(x + unit, y, unit, subHeight,
                 radius=4, fill=LAYER_FILL, stroke=LAYER_STROKE))
        self.addInput(x + unit, y + 0.5 * subHeight)

        yy = y + whiteSpace

        xx = x + unit + 0.5 * (unit - nodeWidth)
        for _ in range(displayCount):
            d.append(drawsvg.Rectangle(xx, yy, nodeWidth,
                     nodeWidth, fill=NODE_FILL, stroke=NODE_STROKE))
            yy += nodeWidth + whiteSpace

        # add bottom
        # register input

        botY = y + subHeight + 0.25 * unit
        botX = x
        self.addInput(botX, botY + 0.5 * subHeight)

        xx = botX + 0.5 * (unit - nodeWidth)

        d.append(drawsvg.Rectangle(botX, botY, unit, subHeight,
                 radius=4, fill=LAYER_FILL, stroke=LAYER_STROKE))
        yy = y + subHeight + 0.25 * unit + whiteSpace
        for _ in range(displayCount):
            d.append(drawsvg.Rectangle(xx, yy, nodeWidth,
                     nodeWidth, fill=NODE_FILL, stroke=NODE_STROKE))
            yy += nodeWidth + whiteSpace

        botX += 1.1 * unit
        d.append(drawsvg.Rectangle(botX, botY, .3 * unit,
                 subHeight, fill=LAYER_FILL, stroke=LAYER_STROKE))

        botX += .4 * unit

        d.append(drawsvg.Rectangle(botX, botY, unit, subHeight,
                 radius=4, fill=LAYER_FILL, stroke=LAYER_STROKE))
        xx += 1.5 * unit
        yy = y + subHeight + 0.25 * unit + whiteSpace
        for _ in range(displayCount):
            d.append(drawsvg.Rectangle(xx, yy, nodeWidth,
                     nodeWidth, fill=NODE_FILL, stroke=NODE_STROKE))
            yy += nodeWidth + whiteSpace

        botX += 1.1 * unit

        d.append(drawsvg.Rectangle(botX, botY, 0.3 * unit,
                 subHeight, fill=LAYER_FILL, stroke=LAYER_STROKE))

        botX += 0.3 * unit

        # add combination bit

        layout_text(d, "add", x + 5.5 * unit, y + 0.5 * (height - subHeight), 0.3 * unit, subHeight)

        topDir = Vector2(x + 2.1 * unit, y + 0.5 * subHeight,
                         x + 5.5 * unit, y + 0.5 * (height - unit))
        d.append(drawsvg.Path(arrow(topDir, 0.5 * unit),
                 fill=ARROW_FILL, stroke="none"))

        bottomDir = Vector2(botX + 0.1 * unit, botY + 0.5 *
                            subHeight, x + 5.5 * unit, y + 0.5 * (height + unit))
        d.append(drawsvg.Path(arrow(bottomDir, 0.5 * unit),
                 fill=ARROW_FILL, stroke="none"))

        font_size = nodeSize * 0.6
        d.append(attachTop(x + 1.5 * unit, y, "R{}{}".format(index,
                                                             "@{}".format(self.stride) if self.stride > 1 else ""),
                           font_size))
        d.append(attachBottom(x + 1.5 * unit, y + height,
                              "{}x{}x{}".format(self.nodeCount, self.outputWidth, self.outputHeight), font_size))


class BatchNormalization(Layer):
    def __init__(self, channels: int, width: int, height: int, transition: Transition) -> None:
        Layer.__init__(self, channels * width * height, transition)
        self.channels = channels
        self.width = width
        self.height = height

    def layout(self, x: float, y: float, width: float, height: float, index: int,
               d: drawsvg.Drawing, allow_full: bool) -> None:
        Layer.layout(self, x, y, width, height, index, d, allow_full)
        layout_text(d, "bn", x, y, width, height)
        Layer.layout(self, x, y, width, height, index, d, allow_full)
        font_size = width * NODE_SIZE_SCALE * 0.6
        d.append(attachTop(x + 0.5 * width, y, "B{}".format(index), font_size))
        d.append(attachBottom(x + 0.5 * width, y + height,
                              "{}x{}x{}".format(self.channels, self.width, self.height), font_size))


class Backbone2D(Layer):
    def __init__(self, net: str, channels: int, width: int, height: int):
        Layer.__init__(self, channels * width * height, Transition.Linear)
        self.net = net
        self.channels = channels
        self.width = width
        self.height = height

    def layout(self, x: float, y: float, width: float, height: float, index: int,
               d: drawsvg.Drawing, allow_full: bool) -> None:
        Layer.layout(self, x, y, width, height, index, d, allow_full)
        layout_text(d, self.net, x, y, width, height)
        font_size = width * NODE_SIZE_SCALE * 0.6
        d.append(attachTop(x + 0.5 * width, y, f"BB{index}", font_size))
        d.append(attachBottom(x + 0.5 * width, y + height, "{}x{}x{}".format(self.channels,
                                                                             self.width, self.height), font_size))


class Interpolate(Layer):
    def __init__(self, op: str, channels: int, width: int, height: int):
        Layer.__init__(self, channels * width * height, Transition.Linear)
        self.op = op
        self.channels = channels
        self.width = width
        self.height = height

    def layout(self, x: float, y: float, width: float, height: float, index: int,
               d: drawsvg.Drawing, allow_full: bool) -> None:
        Layer.layout(self, x, y, width, height, index, d, allow_full)
        layout_text(d, self.op, x, y, width, height)
        font_size = width * NODE_SIZE_SCALE * 0.6
        d.append(attachTop(x + 0.5 * width, y, "I{}".format(index), font_size))
        d.append(attachBottom(x + 0.5 * width, y + height, "{}x{}x{}".format(self.channels,
                                                                             self.width, self.height), font_size))


class Normal(Layer):
    def __init__(self, node_count: int):
        Layer.__init__(self, 0, Transition.Linear, 2)
        self.node_count = node_count
        self.output = node_count

    def layout(self, x: float, y: float, width: float, height: float, index: int,
               d: drawsvg.Drawing, allow_full: bool) -> None:
        unit = width / self.unit_count

        m = y + 0.5 * height
        white_space = 0.5 * unit
        top_y = m - 0.5 * white_space - unit

        # add top (μ)
        layout_text(d, "μ", x, top_y, unit, unit)

        # add bottom (σ)
        bot_y = m + 0.5 * white_space
        layout_text(d, "σ", x, bot_y, unit, unit)

        # register input at center-left
        self.addInput(x, m)

        sample_x = x + 1.2 * unit
        layout_text(d, "sample", sample_x, top_y, 0.8 * unit, 2 * unit + white_space)

        # register output at right edge of sample box
        self.addOutput(sample_x + 0.8 * unit, m)

        label_x = sample_x - .2 * unit
        font_size = unit * NODE_SIZE_SCALE * 0.6
        d.append(attachTop(label_x, top_y, "N{}[{}]".format(index, self.node_count), font_size))
        d.append(attachBottom(label_x, top_y + 2 * unit + white_space, "Normal", font_size))


class RNN(Layer):
    def __init__(self, nodeCount: int, embedSize: int, outputs: Mapping[str, int], transition: Transition) -> None:
        Layer.__init__(self, 2 * nodeCount, transition | Transition.Dropout, 7)
        self.nodeCount = nodeCount
        self.output = nodeCount
        self.embedSize = embedSize
        self.inner_transition = transition
        self.outputs = outputs

    def layout(self, x: float, y: float, width: float, height: float, index: int,
               d: drawsvg.Drawing, allow_full: bool) -> None:
        unit = width / self.unit_count

        nodeSize = unit * NODE_SIZE_SCALE
        subHeight = 0.47 * height - 0.25 * unit
        nodeWidth = 0.7 * nodeSize
        displayCount = math.floor(subHeight / nodeSize)

        whiteSpace = subHeight - (displayCount * nodeWidth)
        whiteSpace /= displayCount + 1

        # add x
        # register input
        layout_linear(d, x, y, unit, subHeight, self.nodeCount)
        self.addInput(x, y + 0.5 * subHeight)

        d.append(attachTop(x + 0.5 * unit, y, f"F_XH[{self.nodeCount}]", nodeSize * 0.5))
        d.append(attachBottom(x + 0.5 * unit, y + subHeight, str(self.nodeCount), nodeSize * 0.4))

        # add h

        botY = y + height - subHeight

        layout_linear(d, x, botY, unit, subHeight, self.nodeCount)
        d.append(attachTop(x + 0.5 * unit, botY, f"F_HH[{self.nodeCount}]", nodeSize * 0.5))
        d.append(attachBottom(x + 0.5 * unit, botY + subHeight, str(self.nodeCount), nodeSize * 0.4))

        # add combination bit

        xx = x + 3 * unit
        layout_text(d, "add", xx, y, unit, height)
        layoutTransition(d, Transition.Identity, x + unit, y + 0.5 * subHeight,
                         xx, y + 0.5 * subHeight, nodeSize * 0.5)
        layoutTransition(d, Transition.Identity, x + unit, botY + 0.33 * subHeight,
                         xx, botY + 0.33 * subHeight, nodeSize * 0.5)
        layoutTransition(d, self.inner_transition, xx, botY + 0.66 * subHeight,
                         x + unit, botY + 0.66 * subHeight, nodeSize * 0.5)

        # add outputs

        xx = x + 6 * unit
        output_height = 0
        for _, node_count in self.outputs.items():
            output_height += math.log(node_count) * nodeSize

        if len(self.outputs) > 1:
            whiteSpace = (height - output_height) / (len(self.outputs) - 1)
            yy = y
        else:
            whiteSpace = 0
            yy = y + 0.5 * (height - output_height)

        for name, node_count in self.outputs.items():
            subHeight = math.log(node_count) * nodeSize
            layout_linear(d, xx, yy, unit, subHeight, node_count)
            d.append(attachTop(xx + 0.5 * unit, yy, f"F_{name}[{node_count}]", nodeSize * 0.5))
            d.append(attachBottom(xx + 0.5 * unit, yy + subHeight, str(node_count), nodeSize * 0.4))
            layoutTransition(d, self.transition, x + 4 * unit, yy + 0.5 * subHeight,
                             xx, yy + 0.5 * subHeight, nodeSize * 0.6)
            yy += subHeight + whiteSpace


class InputTuple(Layer):
    def __init__(self, names: List[str], node_counts: List[int], transition: Transition):
        self.layers: List[Layer] = []
        num_outputs = 0
        for name, node_count in zip(names, node_counts):
            self.layers.append(Input(name, node_count, transition))
            num_outputs += node_count
        Layer.__init__(self, num_outputs, transition)

    def measure(self, unit_width: float, network_height: float, max_height: float,
                max_outputs: int, allow_full: bool) -> Size:
        max_width = 0
        total_height = 0
        for layer in self.layers:
            size = layer.measure(unit_width, network_height, max_height, max_outputs, allow_full)
            if size.width > max_width:
                max_width = size.width

            total_height += size.height

        return Size(max_width, min(max_height, 1.25 * total_height))

    def layout(self, x: float, y: float, width: float, height: float, index: int,
               d: drawsvg.Drawing, allow_full: bool) -> None:
        whitespace = 0.2 * height
        y_space = whitespace / (len(self.layers) - 1)
        input_height = (height - whitespace) / len(self.layers)
        yy = y
        for layer in self.layers:
            layer.layout(x, yy, width, input_height, index, d, allow_full)
            yy += input_height + y_space
            self.addOutput(layer.outputs[0])


class TransformerBlock(Layer):
    def __init__(self, nodeCount: int, numBlocks: int, num_heads: int, decoder: bool,
                 transition: Transition, masked=False):
        unitCount = 20 if decoder else 14
        Layer.__init__(self, 0, transition, unitCount)
        self.decoder = decoder
        self.masked = masked
        self.nodeCount = nodeCount
        self.numBlocks = numBlocks
        self.num_heads = num_heads
        self.output = nodeCount * 4

    def layout(self, x: float, y: float, width: float, height: float, index: int,
               d: drawsvg.Drawing, allow_full: bool) -> None:
        unit = width / self.unit_count

        nodeSize = unit * NODE_SIZE_SCALE
        nodeWidth = 0.7 * nodeSize
        subHeight = 0.7 * height
        displayCount = math.floor(subHeight / nodeSize)

        ySpace = 0
        whiteSpace = subHeight - (displayCount * nodeWidth)
        whiteSpace /= displayCount + 1

        yy = y + ySpace

        xx = x

        font_size = nodeSize

        xPadding = 0.6 * nodeSize
        yPadding = 0.8 * nodeSize
        d.append(drawsvg.Rectangle(xx - xPadding, yy - yPadding, width + 2*xPadding, height + 2*yPadding,
                                   fill="white", stroke=ACTION_FILL,
                                   stroke_dasharray=f"{nodeSize * 0.5} {nodeSize * 0.4}"))
        centerX = x + 0.45 * width if self.decoder else x + 0.65 * width
        d.append(drawsvg.Text("x {}".format(self.numBlocks), font_size, centerX, y + 0.6 * font_size,
                              fill=ACTION_FILL, font_family=FONT_FAMILY, text_anchor="middle"))

        font_size = nodeSize * 0.6

        # Positional Encoding

        layout_text(d, "Positional Encoding", xx, yy, unit, height)
        self.addInput(xx, yy + 0.5 * height)

        def add_mhsa(xx, yy, masked=False):
            # Multi-Head attention
            inputSpacing = subHeight / 4

            xx += unit
            yy = y + height - subHeight

            q_input = Point(xx - nodeSize, yy + inputSpacing)
            k_input = Point(xx - nodeSize, yy + 2*inputSpacing)
            v_input = Point(xx - nodeSize, yy + 3*inputSpacing)
            layout_text(d, "Q", xx - nodeSize, q_input.y - 0.5 * nodeSize, nodeSize, nodeSize)
            layout_text(d, "K", xx - nodeSize, k_input.y - 0.5 * nodeSize, nodeSize, nodeSize)
            layout_text(d, "V", xx - nodeSize, v_input.y - 0.5 * nodeSize, nodeSize, nodeSize)
            if masked:
                mhsa_text = f"MaskMHSA x{self.num_heads}"
            else:
                mask = None
                mhsa_text = f"MHSA x{self.num_heads}"

            layout_text(d, mhsa_text, xx, yy, unit, subHeight)

            if masked:
                gradient = drawsvg.LinearGradient(0, 1, 1, 0, gradientUnits='objectBoundingBox')
                gradient.add_stop(0, "#fff", 0)
                gradient.add_stop(0.49, "#fff", 0)
                gradient.add_stop(0.51, "#fff", 1)
                gradient.add_stop(1, "#fff", 1)
                mask = drawsvg.Mask()
                mask.append(drawsvg.Rectangle(xx, yy, unit, subHeight, fill=gradient))
                layout_text(d, mhsa_text, xx, yy, unit, subHeight,
                            fill=NODE_FILL, stroke=LAYER_STROKE,
                            text_fill="#fff", mask=mask)

            xx += unit
            return xx, yy, q_input, k_input, v_input

        def add_addln(xx, yy):
            # Add
            layout_text(d, "add", xx, yy, 0.75 * unit, height)

            # Layer Norm
            xx = xx + 0.75 * unit
            layerNormX = xx + 0.75 * unit
            layout_text(d, "LN", xx, yy, 0.75 * unit, height)

            return xx, yy, layerNormX

        xx, yy, q_input, k_input, v_input = add_mhsa(xx + 2*unit, yy, self.decoder or self.masked)
        layoutTransition(d, Transition.Identity, x + unit, q_input.y,
                         *q_input, 0.5 * nodeSize)
        layoutTransition(d, Transition.Identity, x + unit, k_input.y,
                         *k_input, 0.5 * nodeSize)
        layoutTransition(d, Transition.Identity, x + unit, v_input.y,
                         *v_input, 0.5 * nodeSize)

        layoutTransition(d, Transition.Identity, x + unit, y + 0.15 * height,
                         xx + unit, y + 0.15 * height, 0.5 * nodeSize)
        layoutTransition(d, Transition.Identity, xx, y + 0.65 * height,
                         xx + unit, y + 0.65 * height, 0.5 * nodeSize)

        xx, yy, layerNormX = add_addln(xx + unit, y)

        if self.decoder:
            xx, yy, q_input, k_input, v_input = add_mhsa(xx + 2.25*unit, yy)
            layoutTransition(d, Transition.Identity, layerNormX, y + 0.15 * height,
                             xx + unit, y + 0.15 * height, 0.5 * nodeSize)
            layoutTransition(d, Transition.Identity, xx, y + 0.65 * height,
                             xx + unit, y + 0.65 * height, 0.5 * nodeSize)
            layoutTransition(d, Transition.Identity, layerNormX, q_input.y,
                             *q_input, 0.5 * nodeSize)
            inputX = xx - 2.75*unit
            inputY = k_input.y - 0.5 * nodeSize
            inputWidth = 0.75 * unit
            inputHeight = v_input.y - k_input.y + nodeSize
            layout_text(d, "copy", inputX, inputY, inputWidth, inputHeight, dashed=True)
            self.tokenInput = Point(inputX + 0.5 * inputWidth, inputY + inputHeight)

            xx, yy, layerNormX = add_addln(xx + unit, y)

        # Feed Forward
        xx = xx + 1.75*unit
        yy = y + height - subHeight
        layoutTransition(d, Transition.Identity, layerNormX, y + 0.65 * height,
                         xx, y + 0.65 * height, 0.5 * nodeSize)
        layoutTransition(d, Transition.Identity, xx + 4 * unit, y + 0.65 * height,
                         xx + 5 * unit, y + 0.65 * height, 0.5 * nodeSize)
        d.append(drawsvg.Rectangle(xx, yy, unit, subHeight, rx=4, ry=4,
                                   fill=LAYER_FILL, stroke=LAYER_STROKE))
        d.append(attachBottom(xx + 0.5 * unit, yy + subHeight, str(self.nodeCount), font_size))
        d.append(drawsvg.Rectangle(xx + 3 * unit, yy, unit, subHeight, rx=4, ry=4,
                                   fill=LAYER_FILL, stroke=LAYER_STROKE))
        d.append(attachBottom(xx + 3.5 * unit, yy + subHeight, str(self.nodeCount), font_size))
        layoutTransition(d, Transition.GELU, xx + 1.1 * unit, yy + 0.5 * subHeight,
                         xx + 3*unit, yy + 0.5 * subHeight, 0.5 * nodeSize)
        nodeRadius = 0.5 * nodeWidth
        xx += 0.5 * unit
        yy = yy + ySpace + whiteSpace + nodeRadius
        for _ in range(displayCount):
            d.append(drawsvg.Circle(xx, yy, nodeRadius,
                     fill=NODE_FILL, stroke=NODE_STROKE))
            d.append(drawsvg.Circle(xx + 3 * unit, yy, nodeRadius,
                     fill=NODE_FILL, stroke=NODE_STROKE))
            yy += nodeWidth + whiteSpace

        xx = xx + 4.5 * unit
        layoutTransition(d, Transition.Identity, layerNormX, y + 0.15 * height,
                         xx, y + 0.15 * height, 0.5 * nodeSize)
        _, _, layerNormX = add_addln(xx, y)
        self.addOutput(x + width, y + 0.5 * height)
        if not self.decoder:
            self.tokenOutput = Point(layerNormX - 0.375 * unit, y)


class ScaledDotProductAttention(Layer):
    def __init__(self, token_size: int, masked: bool):
        self.masked = masked
        self.token_size = token_size
        Layer.__init__(self, token_size * 3, Transition.Token, 7)

    def measure(self, unit_width: float, network_height: float, max_height: float,
                max_outputs: int, allow_full: bool) -> Size:
        linear_height = max_height * math.log(self.token_size) / math.log(max_outputs)
        return Size(unit_width * self.unit_count, min(3 * linear_height, max_height))

    def layout(self, x: float, y: float, width: float, height: float, index: int,
               d: drawsvg.Drawing, allow_full: bool) -> None:
        unit = width / self.unit_count
        node_size = unit * NODE_SIZE_SCALE
        linearHeight = 0.75 * height / 3
        whitespace = 0.25 * height
        y_space = whitespace / 2

        topSize = unit * NODE_SIZE_SCALE * 0.6
        bottomSize = unit * NODE_SIZE_SCALE * 0.5

        xx = x
        yy = y
        layout_linear(d, xx, yy, unit, linearHeight, self.token_size)
        d.append(attachTop(xx + 0.5 * unit, yy, f"F_Q[{self.token_size}]", topSize))
        d.append(attachBottom(xx + 0.5 * unit, yy + linearHeight, str(self.token_size), bottomSize))
        self.addInput(xx, yy + 0.5 * linearHeight)
        yy += linearHeight + y_space
        layout_linear(d, xx, yy, unit, linearHeight, self.token_size)
        d.append(attachTop(xx + 0.5 * unit, yy, f"F_K[{self.token_size}]", topSize))
        d.append(attachBottom(xx + 0.5 * unit, yy + linearHeight, str(self.token_size), bottomSize))
        self.addInput(xx, yy + 0.5 * linearHeight)
        yy += linearHeight + y_space
        layout_linear(d, xx, yy, unit, linearHeight, self.token_size)
        d.append(attachTop(xx + 0.5 * unit, yy, f"F_V[{self.token_size}]", topSize))
        d.append(attachBottom(xx + 0.5 * unit, yy + linearHeight, str(self.token_size), bottomSize))
        self.addInput(xx, yy + 0.5 * linearHeight)
        layoutTransition(d, Transition.Identity, xx + unit, y + height - linearHeight / 2,
                         xx + 6 * unit, y + height - linearHeight / 2, 0.5 * node_size)

        xx += 2 * unit
        yy = y

        top_height = linearHeight * 2 + y_space
        layout_text(d, "matmul", xx, yy, unit, top_height)
        layoutTransition(d, Transition.Identity,
                         xx - unit, y + 0.5 * linearHeight,
                         xx, y + 0.5 * linearHeight, 0.5 * node_size)
        layoutTransition(d, Transition.Identity,
                         xx - unit, y + 0.5 * height,
                         xx, y + 0.5 * height, 0.5 * node_size)

        xx += 2 * unit
        layout_text(d, "softmax", xx, yy, unit, top_height)
        if self.masked:
            gradient = drawsvg.LinearGradient(0, 1, 1, 0, gradientUnits='objectBoundingBox')
            gradient.add_stop(0, "#fff", 0)
            gradient.add_stop(0.49, "#fff", 0)
            gradient.add_stop(0.51, "#fff", 1)
            gradient.add_stop(1, "#fff", 1)
            mask = drawsvg.Mask()
            mask.append(drawsvg.Rectangle(xx, yy, unit, top_height, fill=gradient))
            layout_text(d, "softmax", xx, yy, unit, top_height,
                        fill=NODE_FILL, stroke=LAYER_STROKE,
                        text_fill="#fff", mask=mask)

        layoutTransition(d, Transition.Identity,
                         xx-unit, y + height/3,
                         xx, y + height/3, 0.5 * node_size)

        xx += 2 * unit
        layoutTransition(d, Transition.Identity,
                         xx-unit, y + height/3,
                         xx, y + height/3, 0.5 * node_size)
        layout_text(d, "multiply", xx, yy, unit, height)


class MultiHeadSelfAttention(Layer):
    def __init__(self, token_size: int, num_heads: int, masked: bool):
        self.masked = masked
        self.num_heads = num_heads
        self.token_size = token_size
        num_outputs = 0
        self.heads = []
        for _ in range(num_heads):
            head = ScaledDotProductAttention(token_size, masked)
            self.heads.append(head)
            num_outputs += head.output

        unit_count = self.heads[0].unit_count + 8
        Layer.__init__(self, num_outputs, Transition.Token, unit_count)

    def measure(self, unit_width: float, network_height: float, max_height: float,
                max_outputs: int, allow_full: bool) -> Size:
        height = 0
        for head in self.heads:
            size = head.measure(unit_width, network_height, max_height, max_outputs, allow_full)
            height += size.height

        return Size(unit_width * self.unit_count, min(max_height, height * 1.1))

    def layout(self, x: float, y: float, width: float, height: float, index: int,
               d: drawsvg.Drawing, allow_full: bool) -> None:
        unit = width / self.unit_count
        head_height = 0.9 * height / self.num_heads
        y_space = 0.1 * height / (self.num_heads - 1)

        node_size = unit * NODE_SIZE_SCALE
        split_height = 0.8 * height / 3
        split_space = 0.2 * height / 2
        xx = x
        yy = y
        split_head = 0.48 * split_height
        split_outputs = []
        for _ in range(3):
            self.addInput(xx, yy + 0.5 * split_height)
            layout_text(d, "h0", xx, yy, unit, split_head)
            split_outputs.append(Point(xx + unit, yy + 0.5 * split_head))
            layout_text(d, "h1", xx, yy + split_height - split_head, unit, split_head)
            split_outputs.append(Point(xx + unit, yy + split_height - 0.5 * split_head))
            yy += split_height + split_space

        xx += 5 * unit
        yy = y
        head_width = self.heads[0].unit_count * unit
        for i, head in enumerate(self.heads):
            head.layout(xx, yy, head_width, head_height, index, d, allow_full)
            for j in range(i, len(split_outputs), self.num_heads):
                split = split_outputs[j]
                x2, y2 = head.inputs[j // self.num_heads]
                layoutTransition(d, Transition.Identity, split.x, split.y,
                                 x2, y2, 0.5 * node_size)

            layoutTransition(d, Transition.Identity, xx + head_width, yy + 0.5 * head_height,
                             xx + head_width + 2 * unit, yy + 0.5 * head_height, 0.5 * node_size)
            sep = yy + head_height + y_space * 0.5
            if sep < y + height:
                d.append(drawsvg.Line(xx, sep, xx + head_width, sep, stroke=ACTION_FILL,
                                      stroke_dasharray=f"{unit * 0.5} {unit * 0.4}"))
            yy += head_height + y_space

        xx += head_width + 2 * unit
        yy = y
        layout_text(d, "Concat", xx, yy, unit, height)


class Transformer(Layer):
    def __init__(self, token_size: int, num_blocks: int, num_heads: int, encoder_onehot=None, decoder_onehot=None,
                 no_embed=False):
        self.decoder = TransformerBlock(token_size, num_blocks, num_heads, True, Transition.Token)
        self.encoder = TransformerBlock(token_size, num_blocks, num_heads, False, Transition.Token)
        self.token_size = token_size
        output = self.encoder.output + self.decoder.output
        self.encoder_onehot = OneHot(encoder_onehot, Transition.Token) if encoder_onehot else None
        self.decoder_onehot = OneHot(decoder_onehot, Transition.Token) if decoder_onehot else None
        self.embed = not no_embed
        unitCount = self.decoder.unit_count + 8
        if encoder_onehot:
            unitCount += 2

        if no_embed:
            unitCount -= 2

        Layer.__init__(self, output, Transition.Token, unitCount)

    def measure(self, unit_width: float, network_height: float, max_height: float,
                max_outputs: int, allow_full: bool) -> Size:
        enc_size = self.encoder.measure(unit_width, network_height, max_height, max_outputs, allow_full)
        dec_size = self.decoder.measure(unit_width, network_height, max_height, max_outputs, allow_full)
        return Size(self.unit_count * unit_width,
                    min(1.25 * (enc_size.height + dec_size.height), max_height))

    def layout(self, x: float, y: float, width: float, height: float, index: int,
               d: drawsvg.Drawing, allow_full: bool) -> None:
        enc_width = self.encoder.unit_count * width / self.unit_count
        dec_width = self.decoder.unit_count * width / self.unit_count
        unit = width / self.unit_count

        subHeight = 0.4 * height

        # reduce size of layers below to be the same as the FF linears

        decLeftX = x
        encLeftX = x
        decOffset = 6
        embedHeight = 0.7 * subHeight
        decX = x
        encX = x

        if self.decoder_onehot:
            yy = y + 0.5 * (subHeight - embedHeight)
            self.decoder_onehot.layout(decX, yy, unit, embedHeight, index, d, allow_full)
            decX += 2*unit
            decOffset -= 2
            layoutTransition(d, Transition.Identity, decX - 1 * unit, yy + 0.5 * embedHeight,
                             decX, yy + 0.5 * embedHeight, unit * NODE_SIZE_SCALE * 0.6)

        if self.encoder_onehot:
            yy = y + height - 0.5 * (subHeight + embedHeight)
            self.encoder_onehot.layout(encX, yy, unit, embedHeight, index, d, allow_full)
            encX += 2*unit
            layoutTransition(d, Transition.Identity, encX - 1 * unit, yy + 0.5 * embedHeight,
                             encX, yy + 0.5 * embedHeight, unit * NODE_SIZE_SCALE * 0.6)
            if self.decoder_onehot:
                decOffset += 2

        if self.embed:
            yy = y + 0.5 * (subHeight - embedHeight)
            layout_linear(d, decX, yy, unit, embedHeight, self.token_size)
            d.append(attachBottom(decX + 0.5 * unit, yy + embedHeight,
                     str(self.token_size), unit * NODE_SIZE_SCALE * 0.6))
            decX += 2*unit

            self.addInput(decLeftX, y + 0.2 * height)

            yy = y + height - 0.5 * (subHeight + embedHeight)
            layout_linear(d, encX, yy, unit, embedHeight, self.token_size)
            d.append(attachBottom(encX + 0.5 * unit, yy + embedHeight,
                     str(self.token_size), unit * NODE_SIZE_SCALE * 0.6))
            encX += 2*unit
        else:
            self.addInput(decX + decOffset * unit, y + 0.2 * height)

        yy = y
        self.decoder.layout(decX + decOffset * unit, y, dec_width, subHeight, index, d, allow_full)

        if self.embed:
            layoutTransition(d, Transition.Identity, decX - unit, yy + 0.5 * subHeight,
                             decX + decOffset * unit, yy + 0.5 * subHeight, unit * NODE_SIZE_SCALE * 0.6)

        yy = y + height - subHeight
        self.encoder.layout(encX, yy, enc_width, subHeight, index, d, allow_full)
        layoutTransition(d, Transition.Identity, encX - 1 * unit, yy + 0.5 * subHeight,
                         encX, yy + 0.5 * subHeight, unit * NODE_SIZE_SCALE * 0.6)

        layoutTransition(d, Transition.Token, *self.encoder.tokenOutput,
                         *self.decoder.tokenInput, unit * NODE_SIZE_SCALE * 0.6, True)

        self.addOutput(x + width, y + 0.2 * height)
        self.addInput(encLeftX, y + 0.8 * height)


class TextBox(Layer):
    def __init__(self, text: str, output: int, transition: Transition) -> None:
        Layer.__init__(self, output, transition)
        self.text = text

    def layout(self, x: float, y: float, width: float, height: float, index: int,
               d: drawsvg.Drawing, allow_full: bool) -> None:
        Layer.layout(self, x, y, width, height, index, d, allow_full)
        layout_text(d, self.text, x, y, width, height)
        font_size = width * NODE_SIZE_SCALE * 0.6
        d.append(attachBottom(x + 0.5 * width, y + height,
                              "{}".format(self.output), font_size))


class Concatenate(TextBox):
    def __init__(self, nodeCount: int, transition: Transition) -> None:
        TextBox.__init__(self, "concat", nodeCount, transition)


class PositionalEncoding(TextBox):
    def __init__(self, embed_size: int, transition: Transition) -> None:
        TextBox.__init__(self, "pos enc", embed_size, transition)


class LSTM(Layer):
    def __init__(self, hidden_size: int, cell_size: int):
        self.hidden_size = hidden_size
        self.cell_size = cell_size
        Layer.__init__(self, 4 * cell_size, Transition.Identity, 12)

    def measure(self, unit_width: float, network_height: float, max_height: float,
                max_outputs: int, allow_full: bool) -> Size:
        gate_height = max_height * math.log(self.cell_size) / math.log(max_outputs)
        return Size(unit_width * self.unit_count, min(gate_height * 3.1, max_height))

    def layout(self, x: float, y: float, width: float, height: float, index: int,
               d: drawsvg.Drawing, allow_full: bool) -> None:
        unit = width / self.unit_count
        node_size = unit * NODE_SIZE_SCALE

        xx = x
        yy = y
        layout_text(d, "concat", x, y, unit, height)
        self.addInput(xx, yy + 0.5 * height)

        xx += 2 * unit
        fc_height = 0.7 * height / 4
        fc_space = 0.2 * height / 3
        names = [f"F_ZF[{self.cell_size}]",
                 f"F_ZI[{self.cell_size}]",
                 f"F_ZC[{self.cell_size}]",
                 f"F_ZO[{self.cell_size}]"]

        for name in names:
            layout_linear(d, xx, yy, unit, fc_height, self.hidden_size)
            d.append(attachTop(xx + 0.5 * unit, yy, name, node_size * 0.6))
            d.append(attachBottom(xx + 0.5 * unit, yy + fc_height, str(self.cell_size), node_size * 0.5))
            layoutTransition(d, Transition.Identity, xx - unit, yy + 0.5 * fc_height,
                             xx, yy + 0.5 * fc_height, node_size * 0.5)
            yy += fc_height + fc_space

        xx += 3 * unit
        yy = y
        layout_text(d, "multiply", xx, yy, unit, fc_height)
        layoutTransition(d, Transition.Sigmoid, xx - 2 * unit, y + 0.5 * fc_height,
                         xx, y + 0.5 * fc_height, node_size * 0.5)

        yy = y + fc_height + fc_space
        layout_text(d, "multiply", xx, yy, unit, fc_height * 2 + fc_space)
        layoutTransition(d, Transition.TanH, xx - 2 * unit, y + 1.5 * fc_height + fc_space,
                         xx, y + 1.5 * fc_height + fc_space, node_size * 0.5)
        layoutTransition(d, Transition.Sigmoid, xx - 2 * unit, y + 2.5 * fc_height + 2 * fc_space,
                         xx, y + 2.5 * fc_height + 2 * fc_space, node_size * 0.5)

        xx += 3 * unit
        yy = y
        layout_text(d, "add", xx, yy, unit, 3 * fc_height + 2 * fc_space)
        layoutTransition(d, Transition.Previous, xx, yy + fc_height / 3,
                         xx - 2 * unit, yy + fc_height / 3, node_size * 0.5)
        layoutTransition(d, Transition.Identity, xx - 2 * unit, yy + 2 * fc_height / 3,
                         xx, yy + 2 * fc_height / 3, node_size * 0.5)
        layoutTransition(d, Transition.Identity, xx - 2 * unit, yy + 2 * fc_height + 1.5 * fc_space,
                         xx, yy + 2 * fc_height + 1.5 * fc_space, node_size * 0.5)

        xx += 3 * unit
        yy = y
        layout_text(d, "multiply", xx, yy, unit, height)
        layoutTransition(d, Transition.TanH, xx - 2 * unit, y + 0.5 * (3 * fc_height + 2 * fc_space),
                         xx, y + 0.5 * (3 * fc_height + 2 * fc_space), node_size * 0.5)
        layoutTransition(d, Transition.Sigmoid, x + 3 * unit, yy + 3.5 * fc_height + 3 * fc_space,
                         xx, yy + 3.5 * fc_height + 3 * fc_space, node_size * 0.5)
        layoutTransition(d, Transition.Previous, xx, yy + height - 0.5 * fc_space,
                         x + unit, yy + height - 0.5 * fc_space, unit * 0.5)

        self.addOutput(x + width, y + 0.5 * height)


class LayerDiagram:
    def __init__(self, networkInfo: Mapping, aspectRatio: float, width: float, height: float,
                 line_width: float, outline: bool):
        if line_width > 0:
            num_lines = math.ceil(width / line_width)
            line_height = height
        else:
            line_width = width
            line_height = height
            num_lines = 1

        self.d = drawsvg.Drawing(line_width, line_height * num_lines)
        layerFactory = {
            "FullyConnected": lambda layerInfo: FullyConnected(layerInfo["nodeCount"],
                                                               Transition.create(layerInfo)),
            "Input": lambda layerInfo: Input(layerInfo.get("name", "input"), layerInfo["nodeCount"],
                                             Transition.create(layerInfo)),
            "Convolutional": lambda layerInfo: Convolutional(layerInfo["nodeCount"], layerInfo["width"],
                                                             layerInfo["height"], layerInfo["size"],
                                                             layerInfo.get("stride", 1), layerInfo.get("padding", 0),
                                                             Transition.create(layerInfo)),
            "Input2D": lambda layerInfo: Input2D(layerInfo["name"], layerInfo["channels"], layerInfo["width"],
                                                 layerInfo["height"], Transition.create(layerInfo)),
            "Pooling": lambda layerInfo: Pooling(layerInfo["op"], layerInfo["channels"], layerInfo["width"],
                                                 layerInfo["height"], layerInfo["size"], layerInfo.get("stride", 1),
                                                 layerInfo.get("padding", 0), Transition.create(layerInfo)),
            "ResBlock": lambda layerInfo: ResBlock(layerInfo["nodeCount"], layerInfo["width"], layerInfo["height"],
                                                   layerInfo.get("stride", 1), Transition.create(layerInfo)),
            "BatchNormalization": lambda layerInfo: BatchNormalization(layerInfo["channels"], layerInfo["width"],
                                                                       layerInfo["height"],
                                                                       Transition.create(layerInfo)),
            "Backbone": lambda layerInfo: Backbone2D(layerInfo["net"], layerInfo["channels"], layerInfo["width"],
                                                     layerInfo["height"]),
            "Interpolate": lambda layerInfo: Interpolate(layerInfo["op"], layerInfo["channels"], layerInfo["width"],
                                                         layerInfo["height"]),
            "Normal": lambda layerInfo: Normal(layerInfo["nodeCount"]),
            "ConvolutionalTranspose": lambda layerInfo: ConvolutionalTranspose(layerInfo["nodeCount"],
                                                                               layerInfo["width"],
                                                                               layerInfo["height"],
                                                                               layerInfo["size"],
                                                                               layerInfo.get("stride", 1),
                                                                               layerInfo.get("padding", 0),
                                                                               layerInfo.get("outputPadding", 0),
                                                                               Transition.create(layerInfo)),
            "RNN": lambda layerInfo: RNN(layerInfo["nodeCount"], layerInfo["embedSize"], layerInfo["outputs"],
                                         Transition.create(layerInfo)),
            "OneHot": lambda layerInfo: OneHot(layerInfo["embedSize"], Transition.create(layerInfo)),
            "TransformerBlock": lambda layerInfo: TransformerBlock(layerInfo["token_size"], layerInfo["num_blocks"],
                                                                   layerInfo["num_heads"],
                                                                   decoder=False,
                                                                   masked=layerInfo.get("masked", False),
                                                                   transition=Transition.create(layerInfo)),
            "Concatenate": lambda layerInfo: Concatenate(layerInfo["nodeCount"], Transition.create(layerInfo)),
            "Transformer": lambda layerInfo: Transformer(layerInfo["token_size"], layerInfo["num_blocks"],
                                                         layerInfo["num_heads"],
                                                         layerInfo.get("encoder_onehot", None),
                                                         layerInfo.get("decoder_onehot", None),
                                                         layerInfo.get("no_embed", False)),
            "InputTuple": lambda layerInfo: InputTuple(layerInfo["names"], layerInfo["node_counts"],
                                                       Transition.create(layerInfo)),
            "MultiHeadSelfAttention": lambda layerInfo: MultiHeadSelfAttention(layerInfo["token_size"],
                                                                               layerInfo["num_heads"],
                                                                               layerInfo.get("masked", False)),
            "LSTM": lambda layerInfo: LSTM(layerInfo["hidden_size"], layerInfo["cell_size"]),
            "ScaledDotProductAttention": lambda layerInfo: ScaledDotProductAttention(layerInfo["token_size"],
                                                                                     layerInfo.get("masked", False)),
            "PositionalEncoding": lambda layerInfo: PositionalEncoding(layerInfo["embed_size"],
                                                                       Transition.create(layerInfo))
        }

        self.layers: List[Layer] = []
        layerNames = []
        max_outputs = 0

        for layerInfo in networkInfo["layers"]:
            layer = layerFactory[layerInfo["type"]](layerInfo)
            self.layers.append(layer)
            layerNames.append(layerInfo["name"])
            max_outputs = max(layer.output, max_outputs)

        # layout the network

        if outline:
            self.d.append(drawsvg.Rectangle(0, 0, width, height, fill="none", stroke="red"))

        networkHeight = min(width / aspectRatio, height)

        textSpace = networkHeight / MAX_NODES
        height -= 2 * textSpace
        y = textSpace
        x = textSpace
        width -= 2 * textSpace

        networkHeight = min(width / aspectRatio, height)

        nodeSize = networkHeight / MAX_NODES
        unit_width = nodeSize / NODE_SIZE_SCALE
        unitCount = sum([layer.unit_count for layer in self.layers])
        transitionWidth = (width - (unit_width * unitCount)) / (len(self.layers) - 1)
        transitionHeight = 0.5 * unit_width

        allow_full = all([layer.output <= MAX_NODES for layer in self.layers])
        layer = self.layers[0]
        layer_size = layer.measure(unit_width, networkHeight, height, max_outputs, allow_full)
        layer.layout(x, y + 0.5 * (height - layer_size.height),
                     *layer_size, 0, self.d, allow_full)
        x += layer_size.width + transitionWidth
        ty = y + 0.5 * (height - transitionHeight)

        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            prev = self.layers[i - 1]

            layer_size = layer.measure(unit_width, networkHeight, height, max_outputs, allow_full)

            xdiff = layer.unit_count * unit_width

            line_break = False
            if x + xdiff > line_width:
                x = transitionWidth
                y += line_height
                ty += line_height
                line_break = True

            layer.layout(x, y + 0.5 * (height - layer_size.height),
                         *layer_size, i, self.d, allow_full)

            transitions = []
            starts = prev.outputs
            ends = layer.inputs

            if line_break:
                # Remap start points: wrap to the left edge of the new line
                starts = [Point(0, s.y + line_height) for s in starts]

            if len(starts) == len(ends):
                for start, end in zip(starts, ends):
                    transitions.append((start, end))
            elif len(starts) == 1:
                for end in ends:
                    transitions.append((starts[0], end))
            elif len(ends) == 1:
                for start in starts:
                    transitions.append((start, ends[0]))
            else:
                raise RuntimeError("Invalid number of inputs and outputs")

            for start, end in transitions:
                layoutTransition(self.d, prev.transition, start.x, start.y, end.x, end.y, transitionHeight)

            x += xdiff + transitionWidth


def main():
    global MAX_NODES

    parser = argparse.ArgumentParser("Network Diagram Tool")
    parser.add_argument("network", help="JSON file containing network description")
    parser.add_argument("aspect_ratio", type=float, help="Aspect ratio of diagram")
    parser.add_argument("output", help="Output file path")
    parser.add_argument("--width", type=int, default=420, help="Width of diagram")
    parser.add_argument("--height", type=int, default=150, help="Height of diagram")
    parser.add_argument("--line-width", type=int, default=0)
    parser.add_argument("--outline", action="store_true")
    parser.add_argument("--gray", action="store_true")
    parser.add_argument("--no-labels", action="store_true")
    parser.add_argument("--max-nodes", type=int, default=MAX_NODES)

    args = parser.parse_args()
    with open(args.network) as f:
        networkInfo = json.load(f)

    if args.gray:
        to_grayscale()

    if args.no_labels:
        global LABELS
        LABELS = False

    MAX_NODES = args.max_nodes

    diagram = LayerDiagram(networkInfo, args.aspect_ratio, args.width, args.height, args.line_width, args.outline)

    if args.output.endswith(".svg"):
        diagram.d.save_svg(args.output)
        return

    if args.output.endswith(".pdf"):
        import cairosvg
        _, temppath = tempfile.mkstemp()
        diagram.d.save_svg(temppath)
        cairosvg.svg2pdf(url=temppath, write_to=args.output)
        return

    raise Exception("Unknown output format")


if __name__ == "__main__":
    main()
