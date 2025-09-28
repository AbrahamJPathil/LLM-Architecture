# Context Engineering System

A scalable context engineering system built with Neo4j knowledge graphs, designed for organizational data management and relationship modeling.

## 🎯 Overview

This project implements a comprehensive context engineering solution that leverages **Neo4j graph database** to model complex organizational relationships. The system provides a foundation for building context-aware applications with rich data interconnections.

### Key Features

- 🗂️ **Graph-Based Data Model**: Leverages Neo4j for relationship-rich data storage
- 👥 **Organizational Modeling**: People, projects, tasks, and document management
- 🔗 **Rich Relationships**: Management hierarchies, collaborations, and cross-functional connections  
- 🚀 **Async Operations**: High-performance asynchronous database operations
- 🛡️ **Robust Error Handling**: Comprehensive error management and logging
- 🐳 **Docker Integration**: Easy deployment with containerized Neo4j
- 🎨 **Visual Exploration**: Interactive graph visualization via Neo4j Browser

## 🏗️ Architecture

```
context_engineering/
├── src/
│   ├── knowledge_graph/        # Neo4j client and graph operations
│   │   ├── neo4j_client.py    # Async Neo4j database client
│   │   ├── graph_operations.py # High-level graph operations
│   │   └── query_builder.py   # Cypher query utilities
│   ├── models/                # Data models and schemas
│   │   └── graph_models.py    # Person, Project, Task, Document models
│   └── utils/                 # Utilities and configuration
│       ├── error_handling.py  # Custom exceptions
│       └── logging_config.py  # Logging setup
├── scripts/                   # Data loading and utilities
│   ├── load_sample_data.py   # Sample data loader
│   ├── visualize_data.py     # Graph visualization queries
│   └── clear_data.py         # Database cleanup utilities
├── config/                   # Configuration files
│   └── neo4j_config.yaml    # Neo4j connection settings
├── tests/                    # Unit tests
└── requirements.txt          # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.9+**
- **Docker** (for Neo4j)
- **Git**

### 1. Clone and Setup

```bash
git clone https://github.com/AbrahamJPathil/LLM-Architecture.git
cd context_engineering

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Neo4j Database

```bash
# Start Neo4j with Docker
docker run \
    --name neo4j-context \
    -p 7474:7474 -p 7687:7687 \
    -d \
    -e NEO4J_AUTH=neo4j/test1234 \
    -e NEO4J_PLUGINS=["apoc"] \
    neo4j:latest
```

### 3. Configure Environment

```bash
# Set Neo4j password
export NEO4J_PASSWORD="test1234"  # Linux/Mac
# or
$env:NEO4J_PASSWORD="test1234"   # Windows PowerShell
```

### 4. Load Sample Data

```bash
# Load simple 5-person organizational structure
python scripts/load_sample_data.py
```

### 5. Explore the Graph

Open Neo4j Browser at `http://localhost:7474`
- **Username**: `neo4j`
- **Password**: `test1234`

## 📊 Sample Data Structure

The system creates a simple organizational graph with:

### 👥 **5 People**
- **Alice Johnson** - Executive (Leadership)
- **Bob Smith** - Manager (Engineering)  
- **Carol Davis** - Developer (Engineering)
- **David Wilson** - Designer (Design)
- **Emma Brown** - Developer (Engineering)

### 🔗 **5 Key Relationships**
- Alice **MANAGES** Bob
- Bob **MANAGES** Carol & Emma
- Carol **COLLABORATES_WITH** Emma
- David **WORKS_WITH** Carol

## 🔍 Essential Neo4j Queries

### Basic Exploration
```cypher
# Show all people
MATCH (p:Person) RETURN p

# Show all relationships
MATCH (a)-[r]->(b) RETURN a.name, type(r), b.name

# Complete network visualization
MATCH (n)-[r]-(m) RETURN n, r, m
```

### Management Hierarchy
```cypher
# Management structure
MATCH (manager:Person)-[r:MANAGES]->(employee:Person)
RETURN manager.name as Manager, employee.name as Employee

# Find all managers
MATCH (manager:Person)-[r:MANAGES]->(:Person) 
RETURN manager.name, count(r) as DirectReports
ORDER BY DirectReports DESC
```

### Collaboration Patterns
```cypher
# Cross-department relationships
MATCH (a:Person)-[r]-(b:Person)
WHERE a.department <> b.department
RETURN a.name, a.department, type(r), b.name, b.department

# Find collaboration networks
MATCH (p:Person)-[r:COLLABORATES_WITH|WORKS_WITH]-(colleague:Person)
RETURN p.name, collect(colleague.name) as Collaborators
```

## 🛠️ API Usage

### Basic Operations

```python
import asyncio
from src.knowledge_graph.neo4j_client import Neo4jClient
from src.knowledge_graph.graph_operations import GraphOperations
from src.models.graph_models import PersonNode, UserRole

async def example_usage():
    # Initialize client
    client = Neo4jClient()
    graph_ops = GraphOperations(client)
    
    await client.connect()
    
    # Create a person
    person = PersonNode(
        person_id="emp001",
        name="John Doe",
        email="john@company.com",
        role=UserRole.DEVELOPER,
        department="Engineering",
        hire_date=datetime.now()
    )
    
    person_id = await graph_ops.create_person(person)
    print(f"Created person: {person_id}")
    
    # Find related entities
    related = await graph_ops.find_related_entities(person_id)
    print(f"Found {len(related)} related entities")
    
    await client.close()

# Run example
asyncio.run(example_usage())
```

### Advanced Queries

```python
async def advanced_queries():
    client = Neo4jClient()
    await client.connect()
    
    # Custom Cypher query
    result = await client.execute_query("""
        MATCH (manager:Person)-[:MANAGES*1..3]->(employee:Person)
        RETURN manager.name, collect(employee.name) as Team
    """)
    
    for record in result:
        print(f"{record['manager.name']}: {record['Team']}")
    
    await client.close()
```

## 📁 Data Models

### PersonNode
```python
@dataclass
class PersonNode:
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
```

### Relationship Types
- **MANAGES**: Management hierarchy
- **COLLABORATES_WITH**: Team collaboration
- **WORKS_WITH**: Cross-functional work
- **ASSIGNED_TO**: Task assignments
- **AUTHORED**: Document authorship

## 🧪 Testing

```bash
# Run unit tests
python -m pytest tests/

# Test basic functionality
python main.py

# Visualize current data
python scripts/visualize_data.py
```

## 🐳 Docker Deployment

### Neo4j Configuration
```yaml
# docker-compose.yml
version: '3.8'
services:
  neo4j:
    image: neo4j:latest
    container_name: neo4j-context
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/test1234
      - NEO4J_PLUGINS=["apoc"]
    volumes:
      - ./data:/data
      - ./logs:/logs
```

### Application Deployment
```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "main.py"]
```

## 🔧 Configuration

### Neo4j Settings (`config/neo4j_config.yaml`)
```yaml
neo4j:
  uri: "bolt://localhost:7687"
  username: "neo4j"
  password: "${NEO4J_PASSWORD}"
  database: "neo4j"
  max_connection_lifetime: 3600
  max_connection_pool_size: 50
  connection_acquisition_timeout: 60
```

### Environment Variables
```bash
# Required
NEO4J_PASSWORD=your_neo4j_password

# Optional
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_DATABASE=neo4j
LOG_LEVEL=INFO
```

## 📈 Performance & Scaling

### Database Optimization
- **Indexes**: Automatic creation on `person_id` and other key fields
- **Connection Pooling**: Configurable connection pool management
- **Async Operations**: Non-blocking database operations
- **Query Optimization**: Efficient Cypher query patterns

### Scaling Considerations
- **Horizontal Scaling**: Neo4j Enterprise clustering support
- **Caching**: Application-level caching for frequent queries
- **Batch Operations**: Bulk data loading capabilities
- **Monitoring**: Comprehensive logging and error tracking

## 🛡️ Security

### Authentication
- Neo4j database authentication
- Environment-based credential management
- Secure connection handling

### Data Protection
- Input validation and sanitization
- SQL injection prevention through parameterized queries
- Comprehensive error handling without information leakage

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open Pull Request**

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation for API changes

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Troubleshooting

### Common Issues

**Neo4j Connection Failed**
```bash
# Check if Neo4j is running
docker ps | grep neo4j

# Restart Neo4j container
docker restart neo4j-context
```

**Permission Errors**
```bash
# Ensure environment variable is set
echo $NEO4J_PASSWORD  # Linux/Mac
echo $env:NEO4J_PASSWORD  # Windows
```

**Import Errors**
```bash
# Ensure virtual environment is activated
which python  # Should point to .venv/bin/python

# Reinstall dependencies
pip install -r requirements.txt
```

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python main.py
```

## 📞 Support

- **Documentation**: [Project Wiki](https://github.com/AbrahamJPathil/LLM-Architecture/wiki)
- **Issues**: [GitHub Issues](https://github.com/AbrahamJPathil/LLM-Architecture/issues)
- **Discussions**: [GitHub Discussions](https://github.com/AbrahamJPathil/LLM-Architecture/discussions)

## 🎯 Roadmap

### Current Version (v1.0)
- ✅ Basic graph operations
- ✅ Simple organizational modeling
- ✅ Neo4j integration
- ✅ Docker deployment

### Planned Features (v2.0)
- 🔄 Graphiti knowledge base integration
- 🔄 Grog API implementation
- 🔄 Real-time context updates
- 🔄 Advanced analytics dashboard
- 🔄 REST API endpoints
- 🔄 Authentication system

## 💡 Use Cases

- **HR Management**: Employee relationships and organizational structure
- **Project Management**: Team collaboration and task dependencies  
- **Knowledge Management**: Document relationships and expertise mapping
- **Social Network Analysis**: Communication patterns and influence mapping
- **Context-Aware Applications**: Dynamic context retrieval for AI systems

---

**Built with ❤️ for scalable context engineering**

> *"Context is the key to understanding. Relationships are the foundation of knowledge."*
