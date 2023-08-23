from typing import Union, Optional, NamedTuple
from datetime import datetime, timezone
from dataclasses import dataclass

import flatbuffers
import numpy as np
from streaming_data_types.fbschemas.arrays_wav_00.WaveFormArray import WaveFormArray
from streaming_data_types.fbschemas.arrays_wav_00.DType import DType
from streaming_data_types.fbschemas.arrays_wav_00.WaveFormArray import (
    WaveFormArrayStart,
    WaveFormArrayAddXData,
    WaveFormArrayAddYData,
    WaveFormArrayAddXDataType,
    WaveFormArrayAddYDataType,
    WaveFormArrayAddYTimestamp,
    WaveFormArrayAddXTimestamp,
    WaveFormArrayAddYUnit,
    WaveFormArrayAddXUnit,
    WaveFormArrayAddNumberOfElements,
    WaveFormArrayEnd,
)
from streaming_data_types.utils import check_schema_identifier

FILE_IDENTIFIER = b"wa00"


def serialise_wa00(
    values_x_array: np.ndarray,
    values_y_array: np.ndarray,
    y_timestamp: Optional[datetime] = None,
    x_timestamp: Optional[datetime] = None,
    y_unit: Optional[str] = None,
    x_unit: Optional[str] = None,
    number_of_elements: Optional[np.uint32] = None
) -> bytes:
    builder = flatbuffers.Builder(1024)

    type_map = {
        np.dtype("uint8"): DType.uint8,
        np.dtype("int8"): DType.int8,
        np.dtype("uint16"): DType.uint16,
        np.dtype("int16"): DType.int16,
        np.dtype("uint32"): DType.uint32,
        np.dtype("int32"): DType.int32,
        np.dtype("uint64"): DType.uint64,
        np.dtype("int64"): DType.int64,
        np.dtype("float32"): DType.float32,
        np.dtype("float64"): DType.float64,
    }


    datatype_x_array = type_map[values_x_array.dtype]
    datatype_y_array = type_map[values_y_array.dtype]

    # Build data
    x_data_offset = builder.CreateNumpyVector(values_x_array.view(np.uint8))
    y_data_offset = builder.CreateNumpyVector(values_y_array.view(np.uint8))
    if y_unit is not None:
        y_unit_offset = builder.CreateString(y_unit)
    if x_unit is not None:
        x_unit_offset = builder.CreateString(x_unit)

    # Build the buffer
    WaveFormArrayStart(builder)
    WaveFormArrayAddXData(builder, x_data_offset)
    WaveFormArrayAddYData(builder, y_data_offset)
    WaveFormArrayAddXDataType(builder, datatype_x_array)
    WaveFormArrayAddYDataType(builder, datatype_y_array)
    if y_timestamp is not None:
        WaveFormArrayAddYTimestamp(builder, int(y_timestamp.timestamp() * 1e9))
    if x_timestamp is not None:
        WaveFormArrayAddXTimestamp(builder, int(x_timestamp.timestamp() * 1e9))
    if y_unit is not None:
        WaveFormArrayAddYUnit(builder,y_unit_offset)
    if x_unit is not None:
        WaveFormArrayAddXUnit(builder,x_unit_offset)
    if number_of_elements is not None:
        WaveFormArrayAddNumberOfElements(builder, number_of_elements)

    WA_Message = WaveFormArrayEnd(builder)
    builder.Finish(WA_Message, file_identifier=FILE_IDENTIFIER)
    return bytes(builder.Output())

@dataclass
class wa00_t:
    values_x_array: np.ndarray
    values_y_array: np.ndarray
    y_timestamp: Optional[np.uint64] = None
    x_timestamp: Optional[np.uint64] = None
    x_unit: Optional[str] = None
    y_unit: Optional[str] = None
    number_of_elements: Optional[np.uint32] = None




def get_data(raw_data, datatype) -> np.ndarray:
    """
    Converts the data array into the correct type.
    """
    type_map = {
        DType.uint8: np.uint8,
        DType.int8: np.int8,
        DType.uint16: np.uint16,
        DType.int16: np.int16,
        DType.uint32: np.uint32,
        DType.int32: np.int32,
        DType.uint64: np.uint64,
        DType.int64: np.int64,
        DType.float32: np.float32,
        DType.float64: np.float64,
    }
    return raw_data.view(type_map[datatype])


def deserialise_wa00(buffer: Union[bytearray, bytes]) -> wa00_t:
    check_schema_identifier(buffer, FILE_IDENTIFIER)
    waveform_array = WaveFormArray.GetRootAsWaveFormArray(buffer, 0)
    max_time = datetime(
        year=3001, month=1, day=1, hour=0, minute=0, second=0
    ).timestamp()
    y_timestamp = waveform_array.YTimestamp() / 1e9
    x_timestamp = waveform_array.XTimestamp() / 1e9
    y_unit = waveform_array.YUnit()
    x_unit = waveform_array.XUnit()

    if y_timestamp > max_time:
        y_timestamp = max_time
    if x_timestamp > max_time:
        x_timestamp = max_time
    x_array = get_data(waveform_array.XDataAsNumpy(), waveform_array.XDataType())
    y_array = get_data(waveform_array.YDataAsNumpy(), waveform_array.YDataType())
    y_unit = waveform_array.YUnit()
    x_unit = waveform_array.XUnit()
    number_of_elements = waveform_array.NumberOfElements()
    return wa00_t(
        values_x_array=x_array,
        values_y_array=y_array,
        y_timestamp=datetime.fromtimestamp(y_timestamp, tz=timezone.utc),
        x_timestamp=datetime.fromtimestamp(x_timestamp, tz=timezone.utc),
        y_unit=y_unit,
        x_unit=x_unit,
        number_of_elements=number_of_elements,
    )
