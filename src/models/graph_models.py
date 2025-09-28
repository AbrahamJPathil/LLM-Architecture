"""Data models for graph nodes and relationships."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional
import deal
from src.utils.error_handling import ValidationError

class UserRole(Enum):
    """User roles for context personalization."""
    EXECUTIVE = "executive"
    MANAGER = "manager"
    INDIVIDUAL_CONTRIBUTOR = "individual_contributor"
    LEGAL_MANAGER = "legal_manager"
    LEGAL_REVIEWER = "legal_reviewer"
    DEVELOPER = "developer"
    DESIGNER = "designer"
    GUEST = "guest"

class ProjectStatus(Enum):
    """Project status enumeration."""
    PLANNING = "planning"
    ACTIVE = "active"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class TaskStatus(Enum):
    """Task status enumeration."""
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    DONE = "done"
    BLOCKED = "blocked"

class Priority(Enum):
    """Priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class PersonNode:
    """Represents a person in the knowledge graph."""
    person_id: str
    name: str
    email: str
    role: UserRole
    department: str
    hire_date: datetime
    permissions: List[str] = field(default_factory=list)
    phone: Optional[str] = None
    manager_id: Optional[str] = None
    team_ids: List[str] = field(default_factory=list)
    
    @deal.pre(lambda self: len(self.name.strip()) > 0, message="Name cannot be empty")
    @deal.pre(lambda self: "@" in self.email, message="Email must be valid")
    def __post_init__(self):
        """Validate person data after initialization."""
        self.name = self.name.strip()
        self.email = self.email.lower().strip()

@dataclass
class ProjectNode:
    """Represents a project in the knowledge graph."""
    project_id: str
    name: str
    description: str
    status: ProjectStatus
    start_date: datetime
    end_date: Optional[datetime]
    priority: Priority
    budget: Optional[float] = None
    owner_id: Optional[str] = None
    department_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    @deal.pre(lambda self: len(self.name.strip()) > 0, message="Project name cannot be empty")
    @deal.pre(lambda self: self.budget is None or self.budget >= 0, 
              message="Budget must be non-negative")
    def __post_init__(self):
        """Validate project data after initialization."""
        self.name = self.name.strip()
        if self.end_date and self.end_date < self.start_date:
            raise ValidationError("End date cannot be before start date")

@dataclass
class TaskNode:
    """Represents a task in the knowledge graph."""
    task_id: str
    title: str
    description: str
    status: TaskStatus
    priority: Priority
    created_date: datetime
    due_date: Optional[datetime]
    completed_date: Optional[datetime] = None
    estimated_hours: Optional[float] = None
    actual_hours: Optional[float] = None
    assignee_id: Optional[str] = None
    project_id: Optional[str] = None
    parent_task_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    @deal.pre(lambda self: len(self.title.strip()) > 0, message="Task title cannot be empty")
    @deal.pre(lambda self: self.estimated_hours is None or self.estimated_hours >= 0,
              message="Estimated hours must be non-negative")
    @deal.pre(lambda self: self.actual_hours is None or self.actual_hours >= 0,
              message="Actual hours must be non-negative")
    def __post_init__(self):
        """Validate task data after initialization."""
        self.title = self.title.strip()

@dataclass
class DocumentNode:
    """Represents a document in the knowledge graph."""
    document_id: str
    title: str
    content_type: str
    file_path: Optional[str]
    created_date: datetime
    last_modified: datetime
    author_id: str
    size_bytes: Optional[int] = None
    version: str = "1.0"
    tags: List[str] = field(default_factory=list)
    project_ids: List[str] = field(default_factory=list)
    
    @deal.pre(lambda self: len(self.title.strip()) > 0, message="Document title cannot be empty")
    @deal.pre(lambda self: self.size_bytes is None or self.size_bytes >= 0,
              message="File size must be non-negative")
    def __post_init__(self):
        """Validate document data after initialization."""
        self.title = self.title.strip()

@dataclass
class ConversationNode:
    """Represents a conversation in the knowledge graph."""
    conversation_id: str
    title: str
    start_time: datetime
    end_time: Optional[datetime]
    participant_ids: List[str]
    channel: str
    topic: Optional[str] = None
    message_count: int = 0
    project_ids: List[str] = field(default_factory=list)
    
    @deal.pre(lambda self: len(self.participant_ids) > 0, 
              message="Conversation must have at least one participant")
    @deal.pre(lambda self: self.message_count >= 0, 
              message="Message count must be non-negative")
    def __post_init__(self):
        """Validate conversation data after initialization."""
        self.title = self.title.strip()

@dataclass 
class Relationship:
    """Represents a relationship between two nodes."""
    from_id: str
    to_id: str
    relationship_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    created_date: datetime = field(default_factory=datetime.now)
    
    @deal.pre(lambda self: len(self.from_id.strip()) > 0, message="From ID cannot be empty")
    @deal.pre(lambda self: len(self.to_id.strip()) > 0, message="To ID cannot be empty")
    @deal.pre(lambda self: len(self.relationship_type.strip()) > 0, 
              message="Relationship type cannot be empty")
    def __post_init__(self):
        """Validate relationship data after initialization."""
        self.relationship_type = self.relationship_type.upper()