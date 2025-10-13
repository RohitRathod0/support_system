import sqlite3
import hashlib
import json
import redis
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import uuid

@dataclass
class User:
    user_id: str
    created_at: datetime
    last_active: datetime
    preferences: Dict
    interaction_history: List[Dict]
    satisfaction_scores: List[float]
    preferred_language: str = "en"
    timezone: str = "UTC"
    expertise_level: str = "intermediate"
    communication_style: str = "professional"
    
    def to_dict(self):
        return {
            **asdict(self),
            'created_at': self.created_at.isoformat(),
            'last_active': self.last_active.isoformat()
        }

@dataclass
class Session:
    session_id: str
    user_id: str
    started_at: datetime
    last_activity: datetime
    conversation_turns: List[Dict]
    context_window: Dict
    session_metadata: Dict
    is_active: bool = True
    
    def to_dict(self):
        return {
            **asdict(self),
            'started_at': self.started_at.isoformat(),
            'last_activity': self.last_activity.isoformat()
        }
    
    @property
    def expired(self) -> bool:
        return datetime.now() - self.last_activity > timedelta(hours=24)

class AdvancedUserManager:
    """Enhanced user management with persistent identity and session continuity"""
    
    def __init__(self, db_path: str = "users.db", redis_host: str = "localhost"):
        self.db_path = db_path
        self.setup_database()
        
        try:
            self.session_store = redis.Redis(host=redis_host, decode_responses=True)
        except:
            print("Redis not available, using in-memory storage")
            self.session_store = {}
            
        self.user_profiles = {}
    
    def setup_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                identifier_hash TEXT UNIQUE,
                created_at TEXT,
                last_active TEXT,
                preferences TEXT,
                interaction_history TEXT,
                satisfaction_scores TEXT,
                preferred_language TEXT,
                timezone TEXT,
                expertise_level TEXT,
                communication_style TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT,
                started_at TEXT,
                last_activity TEXT,
                conversation_turns TEXT,
                context_window TEXT,
                session_metadata TEXT,
                is_active INTEGER,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_or_create_user(self, identifier: str) -> User:
        """Create or retrieve persistent user"""
        user_hash = hashlib.sha256(identifier.encode()).hexdigest()[:12]
        user_id = f"user_{user_hash}"
        
        if user_id in self.user_profiles:
            user = self.user_profiles[user_id]
            user.last_active = datetime.now()
            self.update_user_in_db(user)
            return user
        
        user = self.load_user_from_db(user_hash, identifier)
        if user:
            self.user_profiles[user_id] = user
            return user
        
        new_user = User(
            user_id=user_id,
            created_at=datetime.now(),
            last_active=datetime.now(),
            preferences={
                "response_style": "detailed",
                "technical_level": "intermediate",
                "notification_preferences": {"email": True, "sms": False}
            },
            interaction_history=[],
            satisfaction_scores=[],
            preferred_language="en",
            timezone="UTC",
            expertise_level="intermediate",
            communication_style="professional"
        )
        
        self.save_user_to_db(new_user, user_hash)
        self.user_profiles[user_id] = new_user
        
        return new_user
    
    def create_session(self, user_id: str, session_metadata: Dict = None) -> Session:
        """Create new session"""
        session = Session(
            session_id=f"sess_{uuid.uuid4().hex[:8]}",
            user_id=user_id,
            started_at=datetime.now(),
            last_activity=datetime.now(),
            conversation_turns=[],
            context_window={},
            session_metadata=session_metadata or {},
            is_active=True
        )
        
        if hasattr(self.session_store, 'setex'):
            self.session_store.setex(
                session.session_id,
                86400,
                json.dumps(session.to_dict())
            )
        else:
            self.session_store[session.session_id] = session
        
        self.save_session_to_db(session)
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve active session"""
        if hasattr(self.session_store, 'get'):
            session_data = self.session_store.get(session_id)
            if session_data:
                session_dict = json.loads(session_data)
                return self.dict_to_session(session_dict)
        else:
            session = self.session_store.get(session_id)
            if session and not session.expired:
                return session
        
        return self.load_session_from_db(session_id)
    
    def update_session_activity(self, session_id: str, new_turn: Dict):
        """Update session with new turn"""
        session = self.get_session(session_id)
        if session:
            session.last_activity = datetime.now()
            session.conversation_turns.append({
                **new_turn,
                "timestamp": datetime.now().isoformat()
            })
            
            if hasattr(self.session_store, 'setex'):
                self.session_store.setex(
                    session_id, 86400, json.dumps(session.to_dict())
                )
            
            self.update_session_in_db(session)
    
    def get_user_conversation_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get complete conversation history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT conversation_turns FROM sessions 
            WHERE user_id = ? AND is_active = 1 
            ORDER BY last_activity DESC 
            LIMIT ?
        ''', (user_id, limit))
        
        results = cursor.fetchall()
        conn.close()
        
        all_conversations = []
        for (turns_json,) in results:
            turns = json.loads(turns_json) if turns_json else []
            all_conversations.extend(turns)
        
        all_conversations.sort(key=lambda x: x.get('timestamp'), reverse=True)
        return all_conversations[:limit]
    
    def update_user_profile(self, user_id: str, interaction_data: Dict):
        """Update user profile from interaction"""
        user = self.user_profiles.get(user_id)
        if not user:
            return
        
        if 'satisfaction_score' in interaction_data:
            user.satisfaction_scores.append(interaction_data['satisfaction_score'])
            user.satisfaction_scores = user.satisfaction_scores[-20:]
        
        if interaction_data.get('feedback') == 'too_technical':
            user.expertise_level = 'beginner'
        elif interaction_data.get('feedback') == 'need_more_detail':
            user.preferences['response_style'] = 'detailed'
        
        user.interaction_history.append({
            'timestamp': datetime.now().isoformat(),
            'query_type': interaction_data.get('query_type'),
            'resolution_time': interaction_data.get('resolution_time'),
            'satisfaction': interaction_data.get('satisfaction_score')
        })
        
        user.interaction_history = user.interaction_history[-50:]
        self.update_user_in_db(user)
    
    def generate_anonymous_id(self, request_data: Dict) -> str:
        """Generate stable anonymous ID"""
        fingerprint_data = [
            request_data.get('ip_address', ''),
            request_data.get('user_agent', ''),
            request_data.get('accept_language', ''),
            request_data.get('screen_resolution', ''),
            request_data.get('timezone', '')
        ]
        
        fingerprint = '_'.join(fingerprint_data)
        return hashlib.sha256(fingerprint.encode()).hexdigest()[:16]
    
    # Database helper methods
    def save_user_to_db(self, user: User, identifier_hash: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO users VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user.user_id, identifier_hash, user.created_at.isoformat(),
            user.last_active.isoformat(), json.dumps(user.preferences),
            json.dumps(user.interaction_history), json.dumps(user.satisfaction_scores),
            user.preferred_language, user.timezone, user.expertise_level,
            user.communication_style
        ))
        
        conn.commit()
        conn.close()
    
    def load_user_from_db(self, user_hash: str, identifier: str) -> Optional[User]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE identifier_hash = ?', (user_hash,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return User(
                user_id=result[0],
                created_at=datetime.fromisoformat(result[2]),
                last_active=datetime.fromisoformat(result[3]),
                preferences=json.loads(result[4]),
                interaction_history=json.loads(result[5]),
                satisfaction_scores=json.loads(result[6]),
                preferred_language=result[7],
                timezone=result[8],
                expertise_level=result[9],
                communication_style=result[10]
            )
        return None
    
    def update_user_in_db(self, user: User):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE users SET 
            last_active = ?, preferences = ?, interaction_history = ?,
            satisfaction_scores = ?, expertise_level = ?, communication_style = ?
            WHERE user_id = ?
        ''', (
            user.last_active.isoformat(), json.dumps(user.preferences),
            json.dumps(user.interaction_history), json.dumps(user.satisfaction_scores),
            user.expertise_level, user.communication_style, user.user_id
        ))
        
        conn.commit()
        conn.close()
    
    def save_session_to_db(self, session: Session):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session.session_id, session.user_id, session.started_at.isoformat(),
            session.last_activity.isoformat(), json.dumps(session.conversation_turns),
            json.dumps(session.context_window), json.dumps(session.session_metadata),
            1 if session.is_active else 0
        ))
        
        conn.commit()
        conn.close()
    
    def load_session_from_db(self, session_id: str) -> Optional[Session]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM sessions WHERE session_id = ? AND is_active = 1', (session_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return Session(
                session_id=result[0],
                user_id=result[1],
                started_at=datetime.fromisoformat(result[2]),
                last_activity=datetime.fromisoformat(result[3]),
                conversation_turns=json.loads(result[4]) if result[4] else [],
                context_window=json.loads(result[5]) if result[5] else {},
                session_metadata=json.loads(result[6]) if result[6] else {},
                is_active=bool(result[7])
            )
        return None
    
    def update_session_in_db(self, session: Session):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE sessions SET 
            last_activity = ?, conversation_turns = ?, context_window = ?
            WHERE session_id = ?
        ''', (
            session.last_activity.isoformat(),
            json.dumps(session.conversation_turns),
            json.dumps(session.context_window),
            session.session_id
        ))
        
        conn.commit()
        conn.close()
    
    def dict_to_session(self, session_dict: Dict) -> Session:
        """Convert dict to Session object"""
        return Session(
            session_id=session_dict['session_id'],
            user_id=session_dict['user_id'],
            started_at=datetime.fromisoformat(session_dict['started_at']),
            last_activity=datetime.fromisoformat(session_dict['last_activity']),
            conversation_turns=session_dict['conversation_turns'],
            context_window=session_dict['context_window'],
            session_metadata=session_dict['session_metadata'],
            is_active=session_dict['is_active']
        )
