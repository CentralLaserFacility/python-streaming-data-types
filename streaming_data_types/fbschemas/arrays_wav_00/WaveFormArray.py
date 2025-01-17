# automatically generated by the FlatBuffers compiler, do not modify

# namespace: 

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class WaveFormArray(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = WaveFormArray()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsWaveFormArray(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def WaveFormArrayBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x77\x61\x30\x30", size_prefixed=size_prefixed)

    # WaveFormArray
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # WaveFormArray
    def Timestamp(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint64Flags, o + self._tab.Pos)
        return 0

    # WaveFormArray
    def XTimestamp(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint64Flags, o + self._tab.Pos)
        return 0

    # WaveFormArray
    def XDataType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # WaveFormArray
    def YDataType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # WaveFormArray
    def XData(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1))
        return 0

    # WaveFormArray
    def XDataAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint8Flags, o)
        return 0

    # WaveFormArray
    def XDataLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # WaveFormArray
    def XDataIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        return o == 0

    # WaveFormArray
    def YData(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1))
        return 0

    # WaveFormArray
    def YDataAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint8Flags, o)
        return 0

    # WaveFormArray
    def YDataLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # WaveFormArray
    def YDataIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        return o == 0

    # WaveFormArray
    def XUnit(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # WaveFormArray
    def YUnit(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

def WaveFormArrayStart(builder): builder.StartObject(8)
def Start(builder):
    return WaveFormArrayStart(builder)
def WaveFormArrayAddTimestamp(builder, timestamp): builder.PrependUint64Slot(0, timestamp, 0)
def AddTimestamp(builder, timestamp):
    return WaveFormArrayAddTimestamp(builder, timestamp)
def WaveFormArrayAddXTimestamp(builder, xTimestamp): builder.PrependUint64Slot(1, xTimestamp, 0)
def AddXTimestamp(builder, xTimestamp):
    return WaveFormArrayAddXTimestamp(builder, xTimestamp)
def WaveFormArrayAddXDataType(builder, xDataType): builder.PrependInt8Slot(2, xDataType, 0)
def AddXDataType(builder, xDataType):
    return WaveFormArrayAddXDataType(builder, xDataType)
def WaveFormArrayAddYDataType(builder, yDataType): builder.PrependInt8Slot(3, yDataType, 0)
def AddYDataType(builder, yDataType):
    return WaveFormArrayAddYDataType(builder, yDataType)
def WaveFormArrayAddXData(builder, xData): builder.PrependUOffsetTRelativeSlot(4, flatbuffers.number_types.UOffsetTFlags.py_type(xData), 0)
def AddXData(builder, xData):
    return WaveFormArrayAddXData(builder, xData)
def WaveFormArrayStartXDataVector(builder, numElems): return builder.StartVector(1, numElems, 1)
def StartXDataVector(builder, numElems):
    return WaveFormArrayStartXDataVector(builder, numElems)
def WaveFormArrayAddYData(builder, yData): builder.PrependUOffsetTRelativeSlot(5, flatbuffers.number_types.UOffsetTFlags.py_type(yData), 0)
def AddYData(builder, yData):
    return WaveFormArrayAddYData(builder, yData)
def WaveFormArrayStartYDataVector(builder, numElems): return builder.StartVector(1, numElems, 1)
def StartYDataVector(builder, numElems):
    return WaveFormArrayStartYDataVector(builder, numElems)
def WaveFormArrayAddXUnit(builder, xUnit): builder.PrependUOffsetTRelativeSlot(6, flatbuffers.number_types.UOffsetTFlags.py_type(xUnit), 0)
def AddXUnit(builder, xUnit):
    return WaveFormArrayAddXUnit(builder, xUnit)
def WaveFormArrayAddYUnit(builder, yUnit): builder.PrependUOffsetTRelativeSlot(7, flatbuffers.number_types.UOffsetTFlags.py_type(yUnit), 0)
def AddYUnit(builder, yUnit):
    return WaveFormArrayAddYUnit(builder, yUnit)
def WaveFormArrayEnd(builder): return builder.EndObject()
def End(builder):
    return WaveFormArrayEnd(builder)