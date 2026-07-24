#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lig_parser — 共享 LIG 二进制文件解析/写入/信号处理模块

从 LigEdit 的 lig_editor.py 分离，供 lig_editor.py / pipeline.py 及
analytics 包下的 trace / cluster / analyse 模块共用。
"""

import os
import sys
import io
import struct
import math
import logging
from decimal import Decimal

import numpy as np
from scipy.signal import butter, filtfilt


# ============================================================================
#                          资源路径
# ============================================================================

def _resource_path(relative_path):
    """获取资源文件的绝对路径，兼容 PyInstaller 打包环境"""
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


# ============================================================================
#                          站点经纬度匹配
# ============================================================================

def load_station_coords(filepath=None):
    if filepath is None:
        filepath = _resource_path('站点经纬度.txt')
    stations = {}
    if not os.path.exists(filepath):
        return stations
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    i = 0
    while i + 1 < len(lines):
        name = lines[i]
        parts = lines[i + 1].split()
        if len(parts) >= 2:
            try:
                lat = float(parts[0])
                lon = float(parts[1])
                stations[name] = (lat, lon)
            except ValueError:
                pass
        i += 2
    return stations


def match_station_name(lat, lon, stations, tolerance=0.02):
    best_name = "UNKNOWN"
    best_dist = tolerance
    for name, (s_lat, s_lon) in stations.items():
        d = max(abs(lat - s_lat), abs(lon - s_lon))
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name


# ============================================================================
#                          lig 文件解析
# ============================================================================

def ReadGPSTimeFromLig(fp):
    GpsTime = {}
    GpsTime['isTimeValid'] = struct.unpack('<b', fp.read(1))[0]
    GpsTime['isTimeConfirm'] = struct.unpack('<b', fp.read(1))[0]
    fp.read(2)
    GpsTime['Year'] = struct.unpack('<i', fp.read(4))[0]
    GpsTime['Month'] = struct.unpack('<i', fp.read(4))[0]
    GpsTime['Day'] = struct.unpack('<i', fp.read(4))[0]
    GpsTime['Hour'] = struct.unpack('<i', fp.read(4))[0]
    GpsTime['Min'] = struct.unpack('<i', fp.read(4))[0]
    GpsTime['Sec'] = struct.unpack('<i', fp.read(4))[0]
    fp.read(4)
    GpsTime['ActPointSec'] = struct.unpack('<d', fp.read(8))[0]
    Time = ('%02d%02d%02d%02d%02d%010.7f' % (
        GpsTime['Year'], GpsTime['Month'], GpsTime['Day'],
        GpsTime['Hour'], GpsTime['Min'],
        GpsTime['Sec'] + GpsTime['ActPointSec']))
    return str(Time)


def ReadPerLigPieceFromLig(fp, version, skip_waveform=False):
    S = {}
    S['version'] = struct.unpack('<i', fp.read(4))[0]
    S['m_numOfDataOfPerCache'] = struct.unpack('<I', fp.read(4))[0]
    S['m_samplingRate'] = struct.unpack('<d', fp.read(8))[0]
    S['m_preTriggerNum'] = struct.unpack('<i', fp.read(4))[0]
    S['m_numOfData'] = struct.unpack('<i', fp.read(4))[0]
    S['m_numOfChannel'] = struct.unpack('<i', fp.read(4))[0]
    S['m_samplingEventID'] = struct.unpack('<i', fp.read(4))[0]
    S['m_stationName'] = struct.unpack('<32c', fp.read(32))[0]
    S['m_stationID'] = struct.unpack('<i', fp.read(4))[0]
    S['m_isCompress'] = struct.unpack('<b', fp.read(1))[0]
    fp.read(3)
    S['m_dataSize'] = struct.unpack('<i', fp.read(4))[0]
    S['m_cacheCount'] = struct.unpack('<i', fp.read(4))[0]
    S['m_cachePlace'] = struct.unpack('<i', fp.read(4))[0]
    S['m_meanLevel'] = struct.unpack('<i', fp.read(4))[0]
    S['m_trigLevelUp'] = struct.unpack('<i', fp.read(4))[0]
    S['m_trigLevelDown'] = struct.unpack('<i', fp.read(4))[0]
    S['m_matchFileCacheCount'] = struct.unpack('<i', fp.read(4))[0]
    fp.read(4)
    S['m_FirstPointTime'] = ReadGPSTimeFromLig(fp)
    S['m_CurrentGPSTime'] = ReadGPSTimeFromLig(fp)
    S['m_GPSCurrentLocationLat'] = struct.unpack('<d', fp.read(8))[0]
    S['m_GPSCurrentLocationLon'] = struct.unpack('<d', fp.read(8))[0]
    S['m_state'] = struct.unpack('<i', fp.read(4))[0]

    if version == 1001:
        S['offset'] = 2048
        S['m_Digital_Range'] = 2048
        S['m_Range'] = 5
    elif version == 2001:
        S['offset'] = struct.unpack('<i', fp.read(4))[0]
        S['m_LightningLocationLat'] = struct.unpack('<d', fp.read(8))[0]
        S['m_LightningLocationLon'] = struct.unpack('<d', fp.read(8))[0]
        S['m_Range'] = struct.unpack('<d', fp.read(8))[0]
        S['m_Digital_Range'] = struct.unpack('<i', fp.read(4))[0]
        S['Reserved'] = struct.unpack('<224b', fp.read(224))
        S['offset'] = 0
        S['m_Range'] = 10
        S['m_Digital_Range'] = 8192
    elif version == 3001:
        S['offset'] = struct.unpack('<i', fp.read(4))[0]
        S['m_LightningLocationLat'] = struct.unpack('<d', fp.read(8))[0]
        S['m_LightningLocationLon'] = struct.unpack('<d', fp.read(8))[0]
        S['m_Range'] = struct.unpack('<d', fp.read(8))[0]
        S['m_Digital_Range'] = struct.unpack('<i', fp.read(4))[0]
        S['Reserved'] = struct.unpack('<224b', fp.read(224))

    fp.read(4)

    if skip_waveform:
        waveform_size = S['m_numOfData'] * S['m_numOfChannel'] * 2
        fp.seek(waveform_size, 1)
    else:
        for i in range(S['m_numOfChannel']):
            cnt = S['m_numOfData']
            S[str(i)] = struct.unpack(f'<{cnt}H', fp.read(cnt * 2))
    return S


def ReadLigFile(FileName, skip_waveform=False):
    """读取 lig 文件全量数据，返回 {time_str: piece_data}

    skip_waveform=True 时跳过波形数据，仅读取元数据，内存占用降低约 95%。
    """
    with open(FileName, 'rb') as fp:
        Data = {}
        version = struct.unpack('<i', fp.read(4))[0]
        NumOfPiece = struct.unpack('<i', fp.read(4))[0]
        fp.read(4)  # firstPieceCacheCount
        fp.read(4)  # lastPieceCacheCount
        fp.read(4)  # FirstPieceCachePlace
        fp.read(4)  # LastPieceCachePlace
        fp.read(4)  # SamplingEventID
        fp.read(4)  # StationID
        ReadGPSTimeFromLig(fp)  # firstPieceTime
        ReadGPSTimeFromLig(fp)  # lastPieceTime
        error_count = 0
        for i in range(NumOfPiece):
            try:
                piece = ReadPerLigPieceFromLig(fp, version, skip_waveform=skip_waveform)
                time_key = str(piece['m_FirstPointTime'])
                Data[time_key] = piece
            except Exception:
                error_count += 1
    if error_count:
        logging.warning("ReadLigFile(%s): %d/%d pieces failed to parse",
                        os.path.basename(FileName), error_count, NumOfPiece)
    return Data


def ReadLigFileWithOffsets(FileName):
    with open(FileName, 'rb') as fp:
        raw_data = fp.read()

    fp = io.BytesIO(raw_data)
    header = {}
    header['version'] = struct.unpack('<i', fp.read(4))[0]
    header['NumOfPiece'] = struct.unpack('<i', fp.read(4))[0]
    header['firstPieceCacheCount'] = struct.unpack('<I', fp.read(4))[0]
    header['lastPieceCacheCount'] = struct.unpack('<I', fp.read(4))[0]
    header['FirstPieceCachePlace'] = struct.unpack('<I', fp.read(4))[0]
    header['LastPieceCachePlace'] = struct.unpack('<I', fp.read(4))[0]
    header['SamplingEventID'] = struct.unpack('<i', fp.read(4))[0]
    header['StationID'] = struct.unpack('<i', fp.read(4))[0]
    header['m_firstPieceTime'] = ReadGPSTimeFromLig(fp)
    header['m_lastPieceTime'] = ReadGPSTimeFromLig(fp)

    header_size = fp.tell()

    pieces = []
    piece_offsets = []
    error_count = 0
    for i in range(header['NumOfPiece']):
        offset_start = fp.tell()
        try:
            piece = ReadPerLigPieceFromLig(fp, header['version'])
            offset_end = fp.tell()
            time_key = str(piece['m_FirstPointTime'])
            pieces.append((time_key, piece))
            piece_offsets.append((offset_start, offset_end))
        except Exception:
            error_count += 1
    if error_count:
        logging.warning("ReadLigFileWithOffsets(%s): %d/%d pieces failed to parse",
                        os.path.basename(FileName), error_count, header['NumOfPiece'])

    return header, pieces, raw_data, piece_offsets, header_size


# ============================================================================
#                          信号处理
# ============================================================================

def ButterFilter(piece):
    fc = 100000       # 截止频率 100 kHz（原 300 kHz，噪声过滤不足）
    fs = 5000000      # 采样率 5 MHz
    order = 4         # 4 阶
    fc_normalized = fc / (fs / 2)
    b, a = butter(order, fc_normalized, btype='low')
    return filtfilt(b, a, piece)


def CutPieceTo16000(piece):
    """截取波形峰值前后共 16000 点"""
    index_max = np.where(piece == piece.max())[0][0]
    if index_max - 4000 < 0:
        begin = 0
        end = min(begin + 16000, len(piece))
    elif index_max + 12000 > len(piece):
        end = len(piece)
        begin = max(end - 16000, 0)
    else:
        begin = index_max - 4000
        end = begin + 16000
    return piece[begin:end]


def compute_final_time(lig_time, piece):
    """从 lig 时间和波形数据计算精确时间

    返回: (final_time_str, filtered_piece, lig_time_str)
    """
    y = CutPieceTo16000(piece - np.mean(piece))
    y = ButterFilter(y)
    y_abs = np.abs(y)
    peak_index = np.argmax(y_abs)
    time_right = Decimal(lig_time)
    time_int = Decimal(10000)
    Time1 = time_right % time_int
    trigger_time = peak_index * 0.0002
    real_time1 = Time1 + Decimal(str(trigger_time)) * Decimal('0.001')
    real_time = f"{real_time1:.7f}"
    final_time = f"{lig_time.split('.')[0]}.{real_time.split('.')[1]}"
    return final_time, y, lig_time


def compute_peak_voltage(piece_data):
    """计算波形的峰值电压 (V)"""
    if '0' not in piece_data:
        return None
    raw = np.array(piece_data['0'], dtype=np.float64)
    offset = piece_data.get('offset', 0)
    digital_range = piece_data.get('m_Digital_Range', 8192)
    voltage_range = piece_data.get('m_Range', 10)
    voltage = (raw - offset) / digital_range * voltage_range
    voltage_centered = voltage - np.mean(voltage)
    filtered = ButterFilter(voltage_centered)
    peak_v = np.max(np.abs(filtered))
    return peak_v


def voltage_from_piece(piece_data):
    """将原始 uint16 采样值转换为电压数组"""
    if '0' not in piece_data:
        return None
    raw = np.array(piece_data['0'], dtype=np.float64)
    offset = piece_data.get('offset', 0)
    digital_range = piece_data.get('m_Digital_Range', 8192)
    voltage_range = piece_data.get('m_Range', 10)
    return (raw - offset) / digital_range * voltage_range


# ============================================================================
#                          lig 文件写入
# ============================================================================

def repacklig(pulse, time_str, lig_head_path):
    """重新打包 lig 片段：Limitbyt 模板 + 16000 点 uint16 波形 + 替换时间戳"""
    try:
        with open(lig_head_path, 'rb') as f:
            lig_head = f.read()
        YMDHMS = [int(time_str[:2]), int(time_str[2:4]), int(time_str[4:6]),
                   int(time_str[6:8]), int(time_str[8:10]), int(float(time_str[10:12]))]
        Sec = float(time_str[12:])
        PieceYMDHMS = struct.pack('<6i4x', *YMDHMS)
        PieceSec = struct.pack('<d', Sec)
        piece = pulse.tolist()
        pieceFile = lig_head + struct.pack('<16000H', *piece)
        pieceFile = pieceFile[:108] + PieceYMDHMS + PieceSec + pieceFile[108 + 36:]
        return pieceFile
    except Exception:
        return None


class PieceWriter:
    """lig 片段写入器：512 条自动分卷"""

    def __init__(self, output_dir, station_name='', lig_file_head_path=None):
        self.output_dir = output_dir
        self.station_name = station_name
        self.lig_file_head_path = lig_file_head_path or _resource_path('LigHead.lig')
        self.current_path = None
        self.current_fp = None
        self.piece_count = 0
        self.root_time = None
        self.file_index = 1
        self.total_written = 0

    def write(self, piece_data, time_key):
        if self.piece_count >= 512 or self.current_fp is None:
            self._start_new_file(time_key)
        self.current_fp.write(piece_data)
        self.piece_count += 1
        self.total_written += 1

    def _start_new_file(self, time_key):
        self.close()
        if self.root_time is None:
            self.root_time = time_key
        if self.file_index == 1:
            if self.station_name:
                filename = f"{self.station_name}_{self.root_time}.lig"
            else:
                filename = f"{self.root_time}.lig"
        else:
            if self.station_name:
                filename = f"{self.station_name}_{self.root_time}_{self.file_index}.lig"
            else:
                filename = f"{self.root_time}_{self.file_index}.lig"
        self.current_path = os.path.join(self.output_dir, filename)
        self.current_fp = open(self.current_path, 'wb')
        if os.path.exists(self.lig_file_head_path):
            with open(self.lig_file_head_path, 'rb') as hf:
                self.current_fp.write(hf.read())
        self.piece_count = 0
        self.file_index += 1

    def close(self):
        if self.current_fp is not None:
            self.current_fp.close()
            self.current_fp = None


# ============================================================================
#                          时间与距离工具
# ============================================================================

def format_time_display(time_str):
    try:
        if len(time_str) < 13:
            return time_str
        yy = int(time_str[0:2])
        year = 2000 + yy if yy < 50 else 1900 + yy
        month = time_str[2:4]
        day = time_str[4:6]
        hour = time_str[6:8]
        minute = time_str[8:10]
        sec_part = time_str[10:]
        return f"{year}-{month}-{day} {hour}:{minute}:{sec_part}"
    except Exception:
        return time_str


def time_classifier_display(time_str):
    try:
        utc_hour = int(time_str[6:8])
        utc_minute = int(time_str[8:10])
        utc_second = float(time_str[10:])
        utc_total_hour = utc_hour + utc_minute / 60 + utc_second / 3600
        beijing_total_hour = utc_total_hour + 8
        if beijing_total_hour >= 24:
            beijing_total_hour -= 24
        if 5.5 <= beijing_total_hour < 19:
            return "白天"
        else:
            return "夜晚"
    except Exception:
        return "未知"


def format_txt_time(txt_time_str):
    """将时间统一为 YYMMDDhhmmss.fffffff（7 位小数）"""
    s = str(txt_time_str).strip()
    if '.' in s:
        ip, fp = s.split('.', 1)
    else:
        ip, fp = s, ''
    if len(ip) < 12:
        ip = ip.zfill(12)
    elif len(ip) > 12:
        ip = ip[:12]
    fp = (fp + '0' * 7)[:7]
    return f"{ip}.{fp}"


def time_str_to_decimal(time_str):
    """将时间字符串转换为 Decimal 以便精确比较"""
    return Decimal(time_str)


def parse_wwlln_time_to_decimal(date_str, time_str):
    """解析 WWLLN 时间到 YYMMDDhhmmss.ffffff 格式 Decimal"""
    try:
        date_str = date_str.strip().rstrip(',')
        time_str = time_str.strip().rstrip(',')
        parts = date_str.split('/')
        if len(parts) != 3:
            return None
        year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
        time_parts = time_str.split(':')
        if len(time_parts) < 3:
            return None
        hour, minute = int(time_parts[0]), int(time_parts[1])
        sec_part = time_parts[2]
        if '.' in sec_part:
            sec_str, usec_str = sec_part.split('.', 1)
            second = int(sec_str)
            usec_str = usec_str[:6].ljust(6, '0')
            decimal_part = f".{usec_str}"
        else:
            second = int(sec_part)
            decimal_part = ".000000"
        year_short = year % 100
        return Decimal(f"{year_short:02d}{month:02d}{day:02d}"
                       f"{hour:02d}{minute:02d}{second:02d}{decimal_part}")
    except Exception:
        return None


def deg2rad(deg):
    return deg * (math.pi / 180.0)


def haversine_distance(lat1, lon1, lat2, lon2):
    """Haversine 公式计算球面距离 (km)"""
    EARTH_RADIUS = 6371.0
    lat1_r, lon1_r = deg2rad(lat1), deg2rad(lon1)
    lat2_r, lon2_r = deg2rad(lat2), deg2rad(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2)
    return EARTH_RADIUS * 2 * math.asin(math.sqrt(min(a, 1.0)))
