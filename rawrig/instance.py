"""Keep repeated launcher clicks from taking a second camera control connection."""

from pathlib import Path
import os

from PySide6.QtCore import QLockFile, QStandardPaths
from PySide6.QtNetwork import QLocalServer, QLocalSocket


class GuiInstance:
    def __init__(self, demo=False):
        runtime = Path(
            QStandardPaths.writableLocation(
                QStandardPaths.StandardLocation.RuntimeLocation
            )
        )
        runtime.mkdir(parents=True, exist_ok=True)
        suffix = "demo" if demo else "cameras"
        self.name = f"raw-rig-{os.getuid()}-{suffix}"
        self.lock = QLockFile(str(runtime / (self.name + ".lock")))
        self.lock.setStaleLockTime(0)
        self.server = QLocalServer()
        self.server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
        self.owned = False

    def acquire(self, activate):
        if not self.lock.tryLock(0):
            client = QLocalSocket()
            client.connectToServer(self.name)
            if client.waitForConnected(1000):
                client.write(b"activate\n")
                client.waitForBytesWritten(1000)
                client.disconnectFromServer()
            return False
        self.owned = True
        # The PID lock has sole ownership; a crashed predecessor may leave a socket.
        QLocalServer.removeServer(self.name)
        if not self.server.listen(self.name):
            self.lock.unlock()
            self.owned = False
            raise RuntimeError(
                "RAW Rig 실행 창 연결을 만들지 못했습니다: " + self.server.errorString()
            )

        def incoming():
            while self.server.hasPendingConnections():
                client = self.server.nextPendingConnection()
                activate()
                client.disconnectFromServer()
                client.deleteLater()

        self.server.newConnection.connect(incoming)
        return True

    def close(self):
        if self.owned:
            self.server.close()
            self.lock.unlock()
            self.owned = False
