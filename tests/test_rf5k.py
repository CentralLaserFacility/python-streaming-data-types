import pytest
from streaming_data_types.forwarder_config_update_rf5k import (
    serialise_rf5k,
    deserialise_rf5k,
    StreamInfo,
)
from streaming_data_types import SERIALISERS, DESERIALISERS
from streaming_data_types.fbschemas.forwarder_config_update_rf5k.UpdateType import (
    UpdateType,
)


class TestEncoder(object):
    def test_serialises_and_deserialises_rf5k_message_correctly(self):
        """
        Round-trip to check what we serialise is what we get back.
        """
        stream_1 = StreamInfo("channel1", "f142", "topic1")
        stream_2 = StreamInfo("channel2", "TdcTime", "topic2")
        original_entry = {
            "config_change": UpdateType.ADD,
            "streams": [stream_1, stream_2],
        }

        buf = serialise_rf5k(**original_entry)
        entry = deserialise_rf5k(buf)

        assert entry.config_change == original_entry["config_change"]
        assert stream_1 in entry.streams
        assert stream_2 in entry.streams

    def test_if_buffer_has_wrong_id_then_throws(self):
        original_entry = {"config_change": UpdateType.REMOVEALL, "streams": []}

        buf = serialise_rf5k(**original_entry)

        # Manually hack the id
        buf = bytearray(buf)
        buf[4:8] = b"1234"

        with pytest.raises(RuntimeError):
            deserialise_rf5k(buf)

    def test_schema_type_is_in_global_serialisers_list(self):
        assert "rf5k" in SERIALISERS
        assert "rf5k" in DESERIALISERS
