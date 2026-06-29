import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { LiveKitRoom, VideoConference } from '@livekit/components-react';
import '@livekit/components-styles';

const VideoReturnSession = ({ customerId, orderId, productCategory, orderValue }) => {
  const [token, setToken] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [status, setStatus] = useState('connecting');
  const [finalAction, setFinalAction] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let isMounted = true;

    const startSession = async () => {
      try {
        const response = await axios.post('/api/return/start-session', {
          customer_id: customerId,
          order_id: orderId,
          product_category: productCategory,
          order_value: orderValue
        });

        if (isMounted) {
          setToken(response.data.token);
          setSessionId(response.data.session_id);
          setStatus('recording');
        }
      } catch (err) {
        if (isMounted) {
          setError('Failed to start session. Please try again.');
          setStatus('error');
        }
      }
    };

    startSession();

    return () => {
      isMounted = false;
    };
  }, [customerId, orderId, productCategory, orderValue]);

  useEffect(() => {
    if (!sessionId || finalAction) return;

    let intervalId;

    const pollStatus = async () => {
      try {
        const response = await axios.get(`/api/return/status/${sessionId}`);
        const action = response.data?.final_action;

        if (action === 'APPROVED' || action === 'REJECTED' || action === 'PENDING_HUMAN') {
          setFinalAction(action);
          setStatus('done');
          clearInterval(intervalId);
        }
      } catch (err) {
        // Gracefully handle 404s until the agent responds
        console.warn('Status check pending or failed:', err);
      }
    };

    intervalId = setInterval(pollStatus, 5000);

    return () => clearInterval(intervalId);
  }, [sessionId, finalAction]);

  return (
    <div style={{ maxWidth: '800px', margin: '0 auto', padding: '20px', fontFamily: 'sans-serif' }}>
      {error && (
        <div style={{ color: 'red', marginBottom: '20px', fontWeight: 'bold' }}>
          {error}
        </div>
      )}

      {status === 'connecting' && (
        <div style={{ fontSize: '18px', marginBottom: '20px' }}>
          Setting up your session...
        </div>
      )}

      {token && (
        <div style={{ height: '70vh', minHeight: '400px', position: 'relative' }}>
          <LiveKitRoom
            serverUrl={import.meta.env.VITE_LIVEKIT_URL}
            token={token}
            video={true}
            audio={false}
            connect={true}
            data-lk-theme="default"
            style={{ height: '100%', display: 'flex', flexDirection: 'column' }}
          >
            <VideoConference style={{ flex: 1 }} />
            <p style={{ textAlign: 'center', marginTop: '15px', fontSize: '16px', fontWeight: 'bold' }}>
              Show your product clearly. Our AI agent is reviewing your return request.
            </p>
          </LiveKitRoom>
        </div>
      )}

      {finalAction && (
        <div
          style={{
            backgroundColor: finalAction === 'APPROVED' ? 'green' : finalAction === 'REJECTED' ? 'red' : 'yellow',
            color: finalAction === 'PENDING_HUMAN' ? 'black' : 'white',
            padding: '20px',
            marginTop: '20px',
            borderRadius: '5px',
            textAlign: 'center',
            fontWeight: 'bold',
            fontSize: '18px'
          }}
        >
          {finalAction === 'APPROVED' && 'Your return has been approved.'}
          {finalAction === 'REJECTED' && 'Your return request was not approved.'}
          {finalAction === 'PENDING_HUMAN' && 'Your case is being reviewed by our team.'}
        </div>
      )}
    </div>
  );
};

export default VideoReturnSession;
