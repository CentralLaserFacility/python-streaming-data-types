# automatically generated by the FlatBuffers compiler, do not modify

# namespace: 

import flatbuffers

class WaveFormArray(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsWaveFormArray(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = WaveFormArray()
        x.Init(buf, n + offset)
        return x

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
    def XDataType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # WaveFormArray
    def YDataType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # WaveFormArray
    def XData(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1))
        return 0

    # WaveFormArray
    def XDataAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint8Flags, o)
        return 0

    # WaveFormArray
    def XDataLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # WaveFormArray
    def YData(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 1))
        return 0

    # WaveFormArray
    def YDataAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint8Flags, o)
        return 0

    # WaveFormArray
    def YDataLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

def WaveFormArrayStart(builder): builder.StartObject(5)
def WaveFormArrayAddTimestamp(builder, timestamp): builder.PrependUint64Slot(0, timestamp, 0)
def WaveFormArrayAddXDataType(builder, xDataType): builder.PrependInt8Slot(1, xDataType, 0)
def WaveFormArrayAddYDataType(builder, yDataType): builder.PrependInt8Slot(2, yDataType, 0)
def WaveFormArrayAddXData(builder, xData): builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(xData), 0)
def WaveFormArrayStartXDataVector(builder, numElems): return builder.StartVector(1, numElems, 1)
def WaveFormArrayAddYData(builder, yData): builder.PrependUOffsetTRelativeSlot(4, flatbuffers.number_types.UOffsetTFlags.py_type(yData), 0)
def WaveFormArrayStartYDataVector(builder, numElems): return builder.StartVector(1, numElems, 1)
def WaveFormArrayEnd(builder): return builder.EndObject()
