"""一键启动脚本"""
import uvicorn

from app.config import settings

if __name__ == "__main__":
    print(f"  {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"  服务地址: http://{settings.HOST}:{settings.PORT}")
    print("  API 文档: http://localhost:8000/docs")
    print("  前端界面: http://localhost:8000/")
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )
