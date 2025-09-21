"""Entrypoint to run the FastAPI app using uvicorn.

Run from the `backend` folder:
    python main.py
"""
import os

import uvicorn

if __name__ == '__main__':
    port = int(os.environ.get('PORT', '8000'))
    uvicorn.run('app.main:app', host='127.0.0.1', port=port, reload=True)
