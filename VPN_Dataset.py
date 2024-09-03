import os
import pickle
import concurrent.futures
from scapy.all import PcapReader
from scapy.layers.inet import TCP
from scapy.layers.dns import DNS
from scapy.layers.l2 import Ether
from scapy.packet import Padding
import matplotlib.pyplot as plt
from logger import logger
from classify import *  # noqa

plt.set_loglevel("info")

pcap_root_path = 'C:/Users/23939/Desktop/ss'

# Maximum processing length per packet
MTU_LENGTH = 1024

# Whether the packet needs to be ignored
def omit_packet(packet):
    # SYN, ACK or FIN flag is 1 and there is no load
    if TCP in packet and (packet.flags & 0x13):
        layers = packet[TCP].payload.layers()
        if not layers or (Padding in layers and len(layers) == 1):
            return True

    # DNS
    if DNS in packet:
        return True

    return False


def make_packets(filename):
    packets = []

    basename = os.path.basename(filename)

    # Setting tags based on file names
    if 'malicious' in basename.lower():
        label = 'abnormal'
    else:
        label = 'normal'

    logger.debug(f"File {basename} labeled as {label}.")  # Add debug to log

    idx = 0
    for packet in PcapReader(filename):
        if omit_packet(packet):
            continue

        idx += 1
        if Ether in packet:
            content = bytes(packet.payload)
        else:
            content = bytes(packet)

        packets.append((label, content))

    logger.debug("finish %s", basename)

    return packets



def get_all_pcap_files(root_path):
    pcap_files = []
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.endswith('.pcap') or filename.endswith('.pcapng'):
                pcap_files.append(os.path.join(dirpath, filename))
    return pcap_files


def main():
    packets = []
    files = get_all_pcap_files(pcap_root_path)

    print(f"Total PCAP files found: {len(files)}")

    futures = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [
            executor.submit(make_packets, filename)
            for filename in files
        ]

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            packets.extend(result)
            print(f"Processed {len(result)} packets from a file.")

    packets_filename = os.path.join(os.path.dirname(__file__), 'data/packets.pickle')
    os.makedirs(os.path.dirname(packets_filename), exist_ok=True)
    with open(packets_filename, 'wb') as file:
        file.write(pickle.dumps(packets))

    print(f"Total packets processed: {len(packets)}")


if __name__ == '__main__':
    main()
