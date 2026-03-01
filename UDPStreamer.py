import socket
import struct

# IMPORTANT: This list must match the order on PC 2 exactly.
MARKER_NAMES = ['LFHD', 'RFHD', 'LBHD', 'RBHD', 'C7', 'T10', 'CLAV', 'STRN', 'RBAK', 'LSHO', 'LUPA', 'LELB', 'LFRM',
                'LWRA', 'LWRB', 'LFIN', 'RSHO', 'RUPA', 'RELB', 'RFRM', 'RWRA', 'RWRB', 'RFIN', 'LASI', 'RASI',
                'LPSI', 'RPSI', 'LTHI', 'LKNE', 'LTIB', 'LANK', 'LHEE', 'LTOE', 'RTHI', 'RKNE', 'RTIB', 'RANK',
                'RHEE', 'RTOE']

# Create a quick lookup dictionary: {"Head": 0, "Neck": 1, ...}
NAME_TO_ID = {name: i for i, name in enumerate(MARKER_NAMES)}


class UDPStreamer:
    def __init__(self, ip, port_llm):
        self.ip = ip
        self.port_llm = port_llm

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.frame_count = 0

    def pack_data(self, frame_number, data_dict):
        packet = struct.pack('<IH', frame_number, len(data_dict))

        for name, coords in data_dict.items():
            if name in NAME_TO_ID:
                marker_id = NAME_TO_ID[name]
                x, y, z = coords

                packet += struct.pack('<Hddd', marker_id, x, y, z)
            else:
                print(f"Warning: Marker '{name}' not found in schema.")
                pass

        return packet

    def send(self, data_dict):
        binary_packet = self.pack_data(self.frame_count, data_dict)
        self.sock.sendto(binary_packet, (self.ip, self.port_llm))
        self.frame_count += 1
