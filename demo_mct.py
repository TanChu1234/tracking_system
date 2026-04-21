#!/usr/bin/env python3
"""
Multi-Camera Tracking (MCT) System - Direct Entry Point

Usage:
    # Load cameras from database (recommended):
    python demo_mct.py --use_db
    
    # Load only specific floors from database:
    python demo_mct.py --use_db --floors "3F,1F"
    
    # Use manual RTSP URLs (legacy mode):
    python demo_mct.py --rtsp1 "rtsp://..." --rtsp2 "rtsp://..."
"""
import os
import sys
import argparse

# Configure PyTorch memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Add paths
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'image2map'))
sys.path.append(os.path.join(os.getcwd(), 'API_Face'))


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Camera Tracking System with Face + Body ReID",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load cameras from database (recommended):
  python demo_mct.py --use_db
  
  # Load only specific floors from database:
  python demo_mct.py --use_db --floors "3F,1F"
  
  # Use manual RTSP URLs (legacy mode):
  python demo_mct.py --rtsp1 "rtsp://..." --rtsp2 "rtsp://..."
"""
    )
    
    # TransReID Configuration
    parser.add_argument("--config_file", default="configs/Market/vit_transreid_stride.yml", 
                        help="path to TransReID config")
    parser.add_argument("--weights", default="weights/vit_transreid_market.pth", 
                        help="path to TransReID weights")
    
    # Database mode
    parser.add_argument("--use_db", action="store_true", 
                        help="Load camera configuration from PostgreSQL database")
    parser.add_argument("--floors", type=str, default=None,
                        help="Comma-separated floor names to load (e.g., '3F,1F'). Only used with --use_db")
    
    # Database connection (optional overrides)
    parser.add_argument("--db_host", type=str, default=None,
                        help="Database host (default: 192.168.210.250)")
    parser.add_argument("--db_port", type=int, default=None,
                        help="Database port (default: 5432)")
    parser.add_argument("--db_name", type=str, default=None,
                        help="Database name (default: camera_ai_db)")
    
    # Manual mode (legacy)
    parser.add_argument("--rtsp1", default="rtsp://developer:Inf2026T1@10.29.98.60:554/cam/realmonitor?channel=1&subtype=00", 
                        help="RTSP URL 1")
    parser.add_argument("--rtsp2", default="rtsp://developer:Inf2026T1@10.29.98.58:554/cam/realmonitor?channel=1&subtype=00", 
                        help="RTSP URL 2")
    parser.add_argument("--rtsp3", default="rtsp://developer:Inf2026T1@10.29.98.57:554/cam/realmonitor?channel=1&subtype=00", 
                        help="RTSP URL 3")
    parser.add_argument("--rtsp4", default="rtsp://developer:Inf2026T1@10.29.98.59:554/cam/realmonitor?channel=1&subtype=00", 
                        help="RTSP URL 4")
    parser.add_argument("--rtsp1T1", default="rtsp://developer:Inf2026T1@10.29.98.52:554/cam/realmonitor?channel=1&subtype=00", 
                        help="RTSP URL for cam1T1")
    
    # Camera Floor Configuration (manual mode)
    parser.add_argument("--floor_cam1", type=int, default=3, help="Floor number for cam1 (default: 3)")
    parser.add_argument("--floor_cam2", type=int, default=3, help="Floor number for cam2 (default: 3)")
    parser.add_argument("--floor_cam3", type=int, default=3, help="Floor number for cam3 (default: 3)")
    parser.add_argument("--floor_cam4", type=int, default=3, help="Floor number for cam4 (default: 3)")
    parser.add_argument("--floor_cam1T1", type=int, default=1, help="Floor number for cam1T1 (default: 1)")
    
    args = parser.parse_args()
    
    # Set environment variables for database connection if provided
    if args.db_host:
        os.environ['DB_HOST'] = args.db_host
    if args.db_port:
        os.environ['DB_PORT'] = str(args.db_port)
    if args.db_name:
        os.environ['DB_NAME'] = args.db_name
    
    # Import and run engine
    from mct.core.engine import run_demo
    run_demo(args)


if __name__ == "__main__":
    main()
