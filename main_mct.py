#!/usr/bin/env python3
"""
Multi-Camera Tracking (MCT) System - Main Entry Point

This is the primary entry point for the MCT system.
It uses modular components from the mct package.

Usage:
    # Load cameras from database (recommended):
    python main_mct.py --use_db
    
    # Load only specific floors from database:
    python main_mct.py --use_db --floors "3F,1F"
    
    # Use manual RTSP URLs (legacy mode):
    python main_mct.py --rtsp1 "rtsp://..." --rtsp2 "rtsp://..."
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
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Camera Tracking System with Face + Body ReID",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load cameras from database (recommended):
  python main_mct.py --use_db
  
  # Load only specific floors from database:
  python main_mct.py --use_db --floors "3F,1F"
  
  # Use manual RTSP URLs (legacy mode):
  python main_mct.py --rtsp1 "rtsp://..." --rtsp2 "rtsp://..."
"""
    )
    
    # TransReID Configuration
    parser.add_argument("--config_file", default="configs/Market/vit_transreid_stride.yml", 
                        help="path to TransReID config")
    parser.add_argument("--weights", default="weights/transformer_120.pth", 
                        help="path to TransReID weights")
    
    # Database mode
    parser.add_argument("--use_db", action="store_true",
                        help="Load camera configuration from database instead of command-line args")
    parser.add_argument("--floors", type=str, default=None,
                        help="Comma-separated list of floors to load (e.g., '3F,1F'). Only used with --use_db")
    
    # Database connection (optional overrides)
    parser.add_argument("--db_host", type=str, default=None,
                        help="Database host (default: from environment/localhost)")
    parser.add_argument("--db_port", type=int, default=None,
                        help="Database port (default: from environment/5432)")
    parser.add_argument("--db_name", type=str, default=None,
                        help="Database name (default: camera_ai_db)")
    
    # Manual mode (legacy)
    parser.add_argument("--rtsp1", default="", help="RTSP URL 1")
    parser.add_argument("--rtsp2", default="", help="RTSP URL 2")
    parser.add_argument("--rtsp3", default="", help="RTSP URL 3")
    parser.add_argument("--rtsp4", default="", help="RTSP URL 4")
    parser.add_argument("--rtsp1T1", default="", help="RTSP URL for cam1T1 (Floor 1)")
    parser.add_argument("--floor_cam1", type=int, default=3, help="Floor number for cam1")
    parser.add_argument("--floor_cam2", type=int, default=3, help="Floor number for cam2")
    parser.add_argument("--floor_cam3", type=int, default=3, help="Floor number for cam3")
    parser.add_argument("--floor_cam4", type=int, default=3, help="Floor number for cam4")
    parser.add_argument("--floor_cam1T1", type=int, default=1, help="Floor number for cam1T1")
    
    args = parser.parse_args()
    
    # Set environment variables for database connection if provided
    if args.db_host:
        os.environ['DB_HOST'] = args.db_host
    if args.db_port:
        os.environ['DB_PORT'] = str(args.db_port)
    if args.db_name:
        os.environ['DB_NAME'] = args.db_name
    
    # Run
    print("="*60)
    print("🚀 MCT System - Multi-Floor Architecture (F1-F7)")
    print("="*60)
    print("ℹ️  Features:")
    print("   - 7 Floors Support (F1-F7) with dynamic mapper initialization")
    print("   - REST API for floor/camera enable/disable management")
    print("   - WebSocket broadcast per floor")
    print("   - Face Recognition + Body ReID")
    print("   - MCT Database Logging")
    print("")
    print("📡 API Endpoints (after startup):")
    print("   GET  /api/status                    - System status")
    print("   POST /api/floors/{id}/enable         - Enable floor")
    print("   POST /api/floors/{id}/disable        - Disable floor")
    print("   POST /api/cameras/{id}/enable        - Enable camera")
    print("   POST /api/cameras/{id}/disable       - Disable camera")
    print("="*60)
    
    # Import and run engine
    from mct.core.engine import run_demo
    run_demo(args)


if __name__ == "__main__":
    main()
