# extract feature from json file

import json
import csv
import os
import numpy as np


def calculate_byte_distribution(packet_data):
    byte_dist = [0] * 256
    total_bytes = 0

    for packet in packet_data:
        if 'b' in packet:
            bytes_value = packet['b']
            if bytes_value < 256:
                byte_dist[bytes_value] += 1
                total_bytes += 1

    if total_bytes > 0:
        byte_dist = [x / total_bytes for x in byte_dist]  # Calculate the frequency distribution

    byte_dist_mean = np.mean(byte_dist)
    byte_dist_std = np.std(byte_dist)

    return byte_dist, byte_dist_mean, byte_dist_std


def calculate_entropy(byte_dist):
    entropy = -sum(x * np.log2(x + 1e-6) for x in byte_dist if x > 0)  # Calculate entropy values to avoid log2(0)
    total_entropy = entropy * len(byte_dist)
    return entropy, total_entropy


def extract_features_from_json(json_file):
    extracted_features = []

    # Read each JSON object in the JSON file line by line
    with open(json_file, 'r') as file:
        for line in file:
            try:
                record = json.loads(line)

                # 提取所需的字段
                feature = {
                    'sa': record.get('sa', None),  # source ip
                    'da': record.get('da', None),  # destination ip
                    'sp': record.get('sp', None),  # source port
                    'dp': record.get('dp', None),  # destination port
                    'pr': record.get('pr', None),  # type of protocol
                    'time_start': record.get('time_start', None),  # Timestamp of the start of the flow
                    'time_end': record.get('time_end', None),  # Timestamp of the end of the flow
                    'duration': None,  # Duration of flow
                    'bytes_out': record.get('bytes_out', None),  # Number of bytes passed out
                    'num_pkts_out': record.get('num_pkts_out', None),  # Number of packets passed out
                    'bytes_in': record.get('bytes_in', None),  # Incoming bytes
                    'num_pkts_in': record.get('num_pkts_in', None),  # incoming packets
                    'packets': record.get('packets', []),  # list of packets
                    'tcp_first_seq': None,
                    'tcp_out_flags': None,
                    'tcp_out_first_window_size': None,
                    'tcp_in_flags': None,
                    'tcp_in_first_window_size': None,
                    'tcp_out_opts': None,
                    'tcp_in_opts': None,
                    'tcp_out_opt_len': None,
                    'tcp_in_opt_len': None,
                    'tls_c_version': None,
                    'tls_s_version': None,
                    'tls_sni': None,
                    'tls_scs': None,
                    'tls_c_extensions': None,
                    'tls_s_extensions': None,
                    'tls_c_key_length': None,
                    'tls_c_key_exchange': None,
                    'tls_c_random': None,
                    'tls_s_random': None,
                    'tls_cs': None,
                    'tls_s_cert': None,
                    'ip_out_ttl': None,
                    'ip_in_ttl': None,
                    'ip_out_id': None,
                    'ip_in_id': None,
                    'probable_os_out': None,
                    'probable_os_in': None,
                    'idp_len_out': None,
                    'idp_len_in': None,
                    'idp_out': None,
                    'idp_in': None,
                    'dns_qn': None,
                    'dns_rc': None,
                    'dns_rr': None,
                    'expire_type': None
                }

                # Extract TCP related features
                if 'tcp' in record:
                    tcp = record['tcp']
                    feature['tcp_first_seq'] = tcp.get('first_seq', None)
                    feature['tcp_out_flags'] = tcp.get('out', {}).get('flags', None)
                    feature['tcp_out_first_window_size'] = tcp.get('out', {}).get('first_window_size', None)
                    feature['tcp_in_flags'] = tcp.get('in', {}).get('flags', None)
                    feature['tcp_in_first_window_size'] = tcp.get('in', {}).get('first_window_size', None)
                    feature['tcp_out_opts'] = json.dumps(tcp.get('out', {}).get('opts', None))
                    feature['tcp_in_opts'] = json.dumps(tcp.get('in', {}).get('opts', None))
                    feature['tcp_out_opt_len'] = tcp.get('out', {}).get('opt_len', None)
                    feature['tcp_in_opt_len'] = tcp.get('in', {}).get('opt_len', None)

                #  Extract TLS related features
                if 'tls' in record:
                    tls = record['tls']
                    feature['tls_c_version'] = tls.get('c_version', None)
                    feature['tls_s_version'] = tls.get('s_version', None)
                    feature['tls_sni'] = tls.get('sni', None)
                    feature['tls_scs'] = tls.get('scs', None)
                    feature['tls_c_extensions'] = json.dumps(tls.get('c_extensions', None))
                    feature['tls_s_extensions'] = json.dumps(tls.get('s_extensions', None))
                    feature['tls_c_key_length'] = tls.get('c_key_length', None)  # Client key length
                    feature['tls_c_key_exchange'] = json.dumps(tls.get('c_key_exchange', None))  # Client key exchange data
                    feature['tls_c_random'] = json.dumps(tls.get('c_random', None))  # client random number
                    feature['tls_s_random'] = json.dumps(tls.get('s_random', None))  # server random number
                    feature['tls_cs'] = json.dumps(tls.get('cs', None))  # Client-supported cipher suites
                    feature['tls_s_cert'] = json.dumps(tls.get('s_cert', None))  # Server certificate information

                # Extract IP related features
                if 'ip' in record:
                    ip = record['ip']
                    feature['ip_out_ttl'] = ip.get('out', {}).get('ttl', None)
                    feature['ip_in_ttl'] = ip.get('in', {}).get('ttl', None)
                    feature['ip_out_id'] = ip.get('out', {}).get('id', None)
                    feature['ip_in_id'] = ip.get('in', {}).get('id', None)

                # Extracting operating system fingerprints
                if 'probable_os' in record:
                    feature['probable_os_out'] = record['probable_os'].get('out', None)
                    feature['probable_os_in'] = record['probable_os'].get('in', None)

                # Extract IDP related features
                feature['idp_len_out'] = record.get('idp_len_out', None)
                feature['idp_len_in'] = record.get('idp_len_in', None)
                feature['idp_out'] = record.get('idp_out', None)
                feature['idp_in'] = record.get('idp_in', None)

                # Extract DNS related features
                if 'dns' in record:
                    dns_queries = record['dns']
                    dns_qn = [query.get('qn', None) for query in dns_queries]
                    dns_rc = [query.get('rc', None) for query in dns_queries]
                    dns_rr = [query.get('rr', None) for query in dns_queries]

                    feature['dns_qn'] = json.dumps(dns_qn)
                    feature['dns_rc'] = json.dumps(dns_rc)
                    feature['dns_rr'] = json.dumps(dns_rr)

                # Extract traffic expiry type
                feature['expire_type'] = record.get('expire_type', None)

                # Calculate Duration
                if feature['time_start'] is not None and feature['time_end'] is not None:
                    feature['duration'] = feature['time_end'] - feature['time_start']

                # Calculate byte distribution and statistics
                byte_dist, byte_dist_mean, byte_dist_std = calculate_byte_distribution(feature['packets'])
                feature['byte_dist'] = byte_dist
                feature['byte_dist_mean'] = byte_dist_mean
                feature['byte_dist_std'] = byte_dist_std

                # Calculate entropy related features
                entropy, total_entropy = calculate_entropy(byte_dist)
                feature['entropy'] = entropy
                feature['total_entropy'] = total_entropy

                extracted_features.append(feature)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")

    return extracted_features


def process_all_json_files_in_directory(directory_path, output_csv_file):
    # Specify column names for CSV files
    fieldnames = ['sa', 'da', 'sp', 'dp', 'pr', 'time_start', 'time_end', 'duration', 'bytes_out', 'num_pkts_out',
                  'bytes_in', 'num_pkts_in', 'packets', 'byte_dist', 'byte_dist_mean', 'byte_dist_std', 'entropy',
                  'total_entropy', 'tcp_first_seq', 'tcp_out_flags', 'tcp_out_first_window_size', 'tcp_in_flags',
                  'tcp_in_first_window_size', 'tcp_out_opts', 'tcp_in_opts', 'tcp_out_opt_len', 'tcp_in_opt_len',
                  'tls_c_version', 'tls_s_version', 'tls_sni', 'tls_scs', 'tls_c_extensions', 'tls_s_extensions',
                  'tls_c_key_length', 'tls_c_key_exchange', 'tls_c_random', 'tls_s_random', 'tls_cs', 'tls_s_cert',
                  'ip_out_ttl', 'ip_in_ttl', 'ip_out_id', 'ip_in_id', 'probable_os_out', 'probable_os_in',
                  'idp_len_out', 'idp_len_in', 'idp_out', 'idp_in', 'dns_qn', 'dns_rc', 'dns_rr', 'expire_type']

    with open(output_csv_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        # Iterate through all files in a directory
        for filename in os.listdir(directory_path):
            if filename.endswith(".json"):
                json_file_path = os.path.join(directory_path, filename)
                features = extract_features_from_json(json_file_path)
                for feature in features:
                    # Converts packets and byte_dist lists to string form for storage in CSV files
                    feature['packets'] = json.dumps(feature['packets'])
                    feature['byte_dist'] = json.dumps(feature['byte_dist'])
                    writer.writerow(feature)


def main():
    directory_path = 'pcap_json'  # JSON file folder path
    output_csv_file = 'extracted_features.csv'

    process_all_json_files_in_directory(directory_path, output_csv_file)

    print(f"Features have been extracted and saved to {output_csv_file}")


if __name__ == "__main__":
    main()
