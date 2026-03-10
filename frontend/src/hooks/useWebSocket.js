import { useCallback, useEffect, useRef, useState } from "react";

export default function useWebSocket(url, { onJsonMessage } = {}) {
  const socketRef = useRef(null);
  const [isConnected, setIsConnected] = useState(false);

  const connect = useCallback(() => {
    if (socketRef.current && socketRef.current.readyState <= 1) {
      return;
    }
    const ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => setIsConnected(true);
    ws.onclose = () => setIsConnected(false);
    ws.onerror = () => setIsConnected(false);
    ws.onmessage = (event) => {
      try {
        const parsed = JSON.parse(event.data);
        onJsonMessage?.(parsed);
      } catch {
        // Ignore non-JSON messages.
      }
    };
    socketRef.current = ws;
  }, [onJsonMessage, url]);

  const disconnect = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.close();
      socketRef.current = null;
    }
    setIsConnected(false);
  }, []);

  const sendBlob = useCallback((blob) => {
    const ws = socketRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      return;
    }
    ws.send(blob);
  }, []);

  const sendJson = useCallback((payload) => {
    const ws = socketRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      return;
    }
    ws.send(JSON.stringify(payload));
  }, []);

  useEffect(() => () => disconnect(), [disconnect]);

  return {
    isConnected,
    connect,
    disconnect,
    sendBlob,
    sendJson
  };
}
