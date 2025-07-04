#!/usr/bin/env python3
"""Test JARVIS v3.0 structure without requiring dependencies."""

from pathlib import Path
import json

def test_file_structure():
    """Test that all expected files exist."""
    base_dir = Path(__file__).parent
    
    required_files = [
        "backend/database/__init__.py",
        "backend/database/config.py", 
        "backend/database/models.py",
        "backend/database/services.py",
        "backend/config/__init__.py",
        "backend/config/settings.py",
        "backend/utils/security.py",
        "backend/utils/memory_db.py",
        "backend/utils/logging_config.py",
        "scripts/init_database.py",
        "scripts/backup.sh",
        "scripts/backup_service.py", 
        "Dockerfile",
        "docker-compose.yml",
        ".env.template",
        ".gitignore"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not (base_dir / file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing files:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    else:
        print("✅ All required files exist!")
        return True

def test_config_structure():
    """Test configuration file structure."""
    try:
        with open(".env.template") as f:
            content = f.read()
        
        required_vars = [
            "JARVIS_API_KEY",
            "DATABASE_URL", 
            "APCA_API_KEY_ID",
            "APCA_API_SECRET_KEY",
            "OLLAMA_HOST",
            "LOG_LEVEL"
        ]
        
        missing_vars = []
        for var in required_vars:
            if var not in content:
                missing_vars.append(var)
        
        if missing_vars:
            print("❌ Missing environment variables in .env.template:")
            for var in missing_vars:
                print(f"  - {var}")
            return False
        else:
            print("✅ All required environment variables documented!")
            return True
            
    except Exception as e:
        print(f"❌ Error reading .env.template: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing JARVIS v3.0 Infrastructure Setup")
    print("=" * 50)
    
    all_passed = True
    
    print("\n📁 Testing file structure...")
    all_passed &= test_file_structure()
    
    print("\n⚙️ Testing configuration structure...")
    all_passed &= test_config_structure()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All infrastructure tests passed!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Copy .env.template to .env and configure")
        print("3. Run: python scripts/init_database.py")
        print("4. Start JARVIS: python backend/main.py")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)