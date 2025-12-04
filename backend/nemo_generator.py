"""
ULTRA HIGH-QUALITY Text Generation Engine
Perfect realistic data generation with advanced patterns
"""

import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import json
import random
from datetime import datetime, timedelta
import torch
import re
from faker import Faker
import string

# Transformers for Qwen model
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers not available. Using fallback generator.")

class NeMoTextGenerator:
    """ULTRA HIGH-QUALITY synthetic data generator - Indistinguishable from real data"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 🔥 Initialize multi-locale Faker for global realistic data
        self.fake = Faker([
            'en_US', 'en_GB', 'en_CA', 'en_AU',  # English variants
            'fr_FR', 'de_DE', 'es_ES', 'it_IT',  # European
            'ja_JP', 'ko_KR', 'zh_CN',           # Asian
            'pt_BR', 'nl_NL', 'sv_SE', 'no_NO'   # Others
        ])
        
        # 🔥 Initialize enhanced data pools
        self._init_enhanced_data_pools()
        
        # 🔥 Pattern recognition for smart field detection
        self._init_field_patterns()
        
        # 🔥 Track used values to ensure uniqueness where needed
        self.used_values = {}
        
        if TRANSFORMERS_AVAILABLE:
            try:
                print(f"Loading {model_name}...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True
                )
                if self.device == "cpu":
                    self.model = self.model.to(self.device)
                self.model.eval()
                print(f"✓ {model_name} loaded on {self.device}")
            except Exception as e:
                print(f"Model load failed: {e}")
                print("Using ULTRA HIGH-QUALITY fallback with advanced Faker")
                self.model = None
    
    def _init_enhanced_data_pools(self):
        """🔥 Initialize ULTRA REALISTIC data pools with real-world distributions"""
        
        # Ultra realistic first names with frequency distribution
        self.first_names_common = [
            "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
            "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
            "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa"
        ]
        
        self.first_names_trendy = [
            "Emma", "Olivia", "Ava", "Isabella", "Sophia", "Mia", "Charlotte", "Amelia",
            "Harper", "Evelyn", "Abigail", "Emily", "Ella", "Scarlett", "Grace", "Chloe",
            "Liam", "Noah", "Oliver", "Elijah", "Lucas", "Mason", "Logan", "Alexander",
            "Ethan", "Jacob", "Benjamin", "Jack", "Henry", "Sebastian", "Aiden", "Matthew"
        ]
        
        # Realistic last names with ethnic diversity
        self.last_names = [
            # Anglo-Saxon
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Wilson",
            "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Thompson",
            "White", "Harris", "Clark", "Lewis", "Robinson", "Walker", "Young", "Allen",
            "King", "Wright", "Scott", "Green", "Baker", "Adams", "Nelson", "Hill",
            # Hispanic
            "Garcia", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Perez",
            "Sanchez", "Ramirez", "Torres", "Flores", "Rivera", "Gomez", "Diaz", "Cruz",
            # Asian
            "Nguyen", "Kim", "Park", "Chen", "Wang", "Li", "Zhang", "Liu", "Yang", "Huang",
            # Other
            "O'Brien", "O'Connor", "Murphy", "Kelly", "Sullivan", "McCarthy", "Ryan"
        ]
        
        # Real company names by industry
        self.companies_tech = [
            "DataSync Systems", "CloudVault Inc", "NeuralNet Solutions", "QuantumByte Corp",
            "CodeForge Technologies", "BitStream Innovations", "CyberCore Labs",
            "TechNova Group", "DigitalPulse LLC", "Synergy Software", "NextGen Computing"
        ]
        
        self.companies_retail = [
            "Urban Outfitters Co", "Metro Mart", "City Center Shops", "Sunset Boutique",
            "Prime Retail Group", "Valley Store Inc", "Coastal Markets", "Parkside Trading"
        ]
        
        self.companies_finance = [
            "Premier Financial Group", "Capital Trust Bank", "Wealth Advisors LLC",
            "Summit Investment Partners", "Secure Assets Inc", "Global Finance Corp"
        ]
        
        # Real product names by category
        self.products_electronics = [
            "ThinkPad X1 Carbon", "MacBook Pro 16\"", "Dell XPS 15", "Surface Laptop 5",
            "iPad Pro 12.9\"", "Samsung Galaxy Tab S8", "iPhone 14 Pro", "Galaxy S23 Ultra",
            "Sony WH-1000XM5", "AirPods Pro 2", "Bose QuietComfort 45", "Logitech MX Master 3S",
            "LG UltraFine 27\"", "Dell U2723DE", "Samsung Odyssey G7", "ASUS ROG Swift"
        ]
        
        self.products_clothing = [
            "Levi's 501 Original Jeans", "Nike Air Max 270", "Adidas Ultraboost 22",
            "Patagonia Down Sweater", "North Face Thermoball Jacket", "Columbia Fleece Pullover",
            "Under Armour Tech Tee", "Hanes Cotton T-Shirt", "Calvin Klein Boxer Briefs"
        ]
        
        # Real US cities with population weights
        self.cities_major = [
            ("New York", "NY", "USA"), ("Los Angeles", "CA", "USA"), ("Chicago", "IL", "USA"),
            ("Houston", "TX", "USA"), ("Phoenix", "AZ", "USA"), ("Philadelphia", "PA", "USA"),
            ("San Antonio", "TX", "USA"), ("San Diego", "CA", "USA"), ("Dallas", "TX", "USA"),
            ("San Jose", "CA", "USA"), ("Austin", "TX", "USA"), ("Jacksonville", "FL", "USA")
        ]
        
        self.cities_medium = [
            ("Seattle", "WA", "USA"), ("Denver", "CO", "USA"), ("Boston", "MA", "USA"),
            ("Portland", "OR", "USA"), ("Nashville", "TN", "USA"), ("Atlanta", "GA", "USA"),
            ("Miami", "FL", "USA"), ("Minneapolis", "MN", "USA"), ("Orlando", "FL", "USA")
        ]
        
        # International cities
        self.cities_international = [
            ("London", "ENG", "UK"), ("Paris", "IDF", "France"), ("Tokyo", "TKY", "Japan"),
            ("Berlin", "BER", "Germany"), ("Sydney", "NSW", "Australia"), ("Toronto", "ON", "Canada"),
            ("Singapore", "SGP", "Singapore"), ("Dubai", "DXB", "UAE"), ("Hong Kong", "HKG", "China"),
            ("Amsterdam", "NH", "Netherlands"), ("Barcelona", "CAT", "Spain"), ("Rome", "LAZ", "Italy")
        ]
        
        # Real department names
        self.departments = [
            "Engineering", "Product Management", "Sales", "Marketing", "Customer Success",
            "Human Resources", "Finance", "Operations", "Legal", "IT Support",
            "Data Science", "Security", "Quality Assurance", "Research & Development",
            "Business Development", "Customer Support", "Design", "Content"
        ]
        
        # Real job titles by department
        self.job_titles = {
            "Engineering": [
                "Software Engineer", "Senior Software Engineer", "Lead Engineer", 
                "Engineering Manager", "Principal Engineer", "Staff Engineer",
                "Backend Engineer", "Frontend Engineer", "Full Stack Engineer", "DevOps Engineer"
            ],
            "Product Management": [
                "Product Manager", "Senior Product Manager", "Product Owner",
                "Principal Product Manager", "VP of Product", "Associate Product Manager"
            ],
            "Sales": [
                "Sales Representative", "Account Executive", "Sales Manager",
                "Senior Account Executive", "VP of Sales", "Business Development Rep"
            ],
            "Marketing": [
                "Marketing Manager", "Content Marketing Manager", "Social Media Manager",
                "SEO Specialist", "Marketing Director", "Brand Manager", "Growth Marketing Manager"
            ]
        }
        
        # Real skills with proficiency levels
        self.skills_technical = [
            "Python", "JavaScript", "TypeScript", "Java", "C++", "Go", "Rust", "Ruby",
            "React", "Vue.js", "Angular", "Node.js", "Django", "Flask", "Spring Boot",
            "PostgreSQL", "MongoDB", "Redis", "MySQL", "AWS", "Azure", "GCP",
            "Docker", "Kubernetes", "Terraform", "Jenkins", "Git", "CI/CD"
        ]
        
        self.skills_soft = [
            "Leadership", "Communication", "Problem Solving", "Team Collaboration",
            "Project Management", "Strategic Planning", "Critical Thinking", "Creativity",
            "Time Management", "Adaptability", "Conflict Resolution", "Negotiation"
        ]
        
        # Transaction statuses with realistic distribution
        self.transaction_statuses = {
            "completed": 0.85,  # 85% success rate
            "pending": 0.10,    # 10% pending
            "failed": 0.03,     # 3% failed
            "cancelled": 0.02   # 2% cancelled
        }
        
        # Email domains with realistic distribution
        self.email_domains_personal = [
            ("gmail.com", 0.45), ("yahoo.com", 0.20), ("outlook.com", 0.15),
            ("hotmail.com", 0.10), ("icloud.com", 0.05), ("protonmail.com", 0.03),
            ("aol.com", 0.02)
        ]
        
        self.email_domains_business = [
            "company.com", "corp.com", "inc.com", "group.com", "tech.io",
            "solutions.com", "consulting.com", "partners.com"
        ]
        
        # Phone number formats by country
        self.phone_formats = {
            "US": ["+1-###-###-####", "(###) ###-####", "###-###-####"],
            "UK": ["+44 #### ######", "+44-####-######"],
            "DE": ["+49 ### #######", "+49-###-#######"],
            "FR": ["+33 # ## ## ## ##", "+33-#-##-##-##-##"]
        }
        
        # Credit card types with realistic distribution
        self.credit_cards = {
            "Visa": 0.50,
            "Mastercard": 0.30,
            "American Express": 0.15,
            "Discover": 0.05
        }
        
        # Product categories with subcategories
        self.categories = {
            "Electronics": ["Computers", "Smartphones", "Audio", "Cameras", "Accessories"],
            "Clothing": ["Men's Wear", "Women's Wear", "Kids", "Shoes", "Accessories"],
            "Home & Garden": ["Furniture", "Decor", "Kitchen", "Tools", "Outdoor"],
            "Sports": ["Fitness", "Outdoor", "Team Sports", "Water Sports", "Winter Sports"],
            "Books": ["Fiction", "Non-Fiction", "Technical", "Educational", "Children's"]
        }
    
    def _init_field_patterns(self):
        """🔥 Initialize smart field pattern detection"""
        
        self.field_patterns = {
            # Name patterns
            'name': ['name', 'full_name', 'fullname', 'customer_name', 'user_name'],
            'first_name': ['first_name', 'firstname', 'fname', 'given_name'],
            'last_name': ['last_name', 'lastname', 'lname', 'surname', 'family_name'],
            
            # Contact patterns
            'email': ['email', 'mail', 'email_address', 'contact_email'],
            'phone': ['phone', 'mobile', 'telephone', 'phone_number', 'contact_number'],
            'address': ['address', 'street', 'street_address', 'location'],
            'city': ['city', 'town', 'municipality'],
            'state': ['state', 'province', 'region', 'prefecture'],
            'country': ['country', 'nation'],
            'zip': ['zip', 'zipcode', 'postal_code', 'postcode'],
            
            # Business patterns
            'company': ['company', 'employer', 'organization', 'business'],
            'job': ['job', 'position', 'title', 'role', 'occupation'],
            'department': ['department', 'dept', 'division', 'team'],
            'salary': ['salary', 'compensation', 'pay', 'wage', 'income'],
            
            # Product patterns
            'product': ['product', 'item', 'article', 'good'],
            'category': ['category', 'type', 'class', 'classification'],
            'price': ['price', 'cost', 'amount', 'value', 'rate'],
            'quantity': ['quantity', 'qty', 'count', 'number', 'stock'],
            
            # ID patterns
            'id': ['id', '_id', 'identifier', 'code', 'reference'],
            'uuid': ['uuid', 'guid'],
            
            # Date patterns
            'date': ['date', 'timestamp', 'time', 'datetime'],
            'created': ['created', 'created_at', 'date_created', 'registered'],
            'updated': ['updated', 'modified', 'last_modified', 'changed'],
            
            # Status patterns
            'status': ['status', 'state', 'condition'],
            'active': ['active', 'enabled', 'is_active', 'is_enabled'],
            
            # Financial patterns
            'transaction': ['transaction', 'payment', 'transfer'],
            'currency': ['currency', 'curr'],
            'card': ['card', 'credit_card', 'payment_method']
        }
    
    def _detect_field_type(self, field_name: str, field_spec: Any) -> str:
        """🔥 Smart field type detection based on name patterns"""
        
        field_lower = field_name.lower()
        
        # Check against patterns
        for pattern_type, patterns in self.field_patterns.items():
            if any(p in field_lower for p in patterns):
                return pattern_type
        
        return 'generic'
    
    async def generate(
        self,
        schema: Dict[str, Any],
        num_samples: int,
        format: str = "json",
        constraints: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Generate ULTRA HIGH-QUALITY synthetic data
        
        Args:
            schema: Data schema definition
            num_samples: Number of samples to generate
            format: Output format (json, csv, sql)
            constraints: Optional constraints
            
        Returns:
            Perfect realistic data
        """
        
        # Reset used values for uniqueness tracking
        self.used_values = {}
        
        # Always use high-quality generator (better than LLM for structured data)
        return await self._generate_ultra_quality(
            schema, num_samples, format, constraints
        )
    
    async def _generate_ultra_quality(
        self,
        schema: Dict[str, Any],
        num_samples: int,
        format: str,
        constraints: Optional[Dict[str, Any]]
    ) -> Any:
        """🔥 Generate ULTRA QUALITY data - Indistinguishable from real"""
        
        samples = []
        
        for i in range(num_samples):
            sample = {}
            
            for field_name, field_spec in schema.items():
                # Parse field specification
                if isinstance(field_spec, dict):
                    field_type = field_spec.get("type", "string")
                    examples = field_spec.get("examples", [])
                    minimum = field_spec.get("minimum")
                    maximum = field_spec.get("maximum")
                    description = field_spec.get("description", "")
                    unique = field_spec.get("unique", False)
                elif isinstance(field_spec, str):
                    field_type = field_spec
                    examples = []
                    minimum = None
                    maximum = None
                    description = ""
                    unique = False
                else:
                    field_type = "string"
                    examples = []
                    minimum = None
                    maximum = None
                    description = ""
                    unique = False
                
                # Generate perfect value
                value = self._generate_perfect_value(
                    field_name=field_name,
                    field_type=field_type,
                    examples=examples,
                    minimum=minimum,
                    maximum=maximum,
                    description=description,
                    unique=unique,
                    constraints=constraints,
                    index=i,
                    current_sample=sample
                )
                
                sample[field_name] = value
            
            samples.append(sample)
        
        return self._format_output(samples, format)
    
    def _generate_perfect_value(
        self,
        field_name: str,
        field_type: str,
        examples: List[Any],
        minimum: Optional[Union[int, float]],
        maximum: Optional[Union[int, float]],
        description: str,
        unique: bool,
        constraints: Optional[Dict],
        index: int,
        current_sample: Dict[str, Any]
    ) -> Any:
        """🔥 Generate PERFECT realistic values"""
        
        field_lower = field_name.lower()
        detected_type = self._detect_field_type(field_name, field_type)
        
        # === NAME FIELDS ===
        if detected_type == 'first_name':
            # 70% common names, 30% trendy names (realistic distribution)
            if random.random() < 0.7:
                return random.choice(self.first_names_common)
            else:
                return random.choice(self.first_names_trendy)
        
        elif detected_type == 'last_name':
            return random.choice(self.last_names)
        
        elif detected_type == 'name':
            first = random.choice(self.first_names_common + self.first_names_trendy)
            last = random.choice(self.last_names)
            
            # Sometimes add middle initial (20% of the time)
            if random.random() < 0.2:
                middle = random.choice(string.ascii_uppercase)
                return f"{first} {middle}. {last}"
            return f"{first} {last}"
        
        # === EMAIL FIELDS ===
        elif detected_type == 'email' or field_type == 'email':
            # Get name from current sample if available
            if 'first_name' in current_sample and 'last_name' in current_sample:
                first = current_sample['first_name'].lower()
                last = current_sample['last_name'].lower()
            elif 'name' in current_sample:
                parts = current_sample['name'].lower().split()
                first = parts[0] if len(parts) > 0 else self.fake.first_name().lower()
                last = parts[-1] if len(parts) > 1 else self.fake.last_name().lower()
            else:
                first = random.choice(self.first_names_common).lower()
                last = random.choice(self.last_names).lower()
            
            # Generate realistic email
            patterns = [
                f"{first}.{last}",           # john.smith
                f"{first}{last}",            # johnsmith
                f"{first}_{last}",           # john_smith
                f"{first}{last[0]}",         # johns
                f"{first[0]}{last}",         # jsmith
                f"{first}.{last}{random.randint(1,99)}"  # john.smith42
            ]
            
            username = random.choice(patterns)
            
            # Choose domain (80% personal, 20% business)
            if random.random() < 0.8:
                # Personal email
                domain = random.choices(
                    [d[0] for d in self.email_domains_personal],
                    weights=[d[1] for d in self.email_domains_personal]
                )[0]
            else:
                # Business email
                if 'company' in current_sample:
                    company_slug = current_sample['company'].lower().split()[0]
                    domain = f"{company_slug}.com"
                else:
                    domain = random.choice(self.email_domains_business)
            
            email = f"{username}@{domain}"
            
            # Ensure uniqueness if required
            if unique:
                counter = 1
                original_email = email
                while email in self.used_values.get(field_name, set()):
                    email = f"{username}{counter}@{domain}"
                    counter += 1
                self.used_values.setdefault(field_name, set()).add(email)
            
            return email
        
        # === PHONE FIELDS ===
        elif detected_type == 'phone':
            country = "US"  # Default to US
            if 'country' in current_sample:
                country_code = current_sample['country']
                if country_code in self.phone_formats:
                    country = country_code
            
            format_pattern = random.choice(self.phone_formats.get(country, self.phone_formats["US"]))
            phone = ""
            for char in format_pattern:
                if char == '#':
                    phone += str(random.randint(0, 9))
                else:
                    phone += char
            return phone
        
        # === ADDRESS FIELDS ===
        elif detected_type == 'address':
            return self.fake.street_address()
        
        elif detected_type == 'city':
            if examples:
                return random.choice(examples)
            # 60% major cities, 30% medium, 10% international
            rand = random.random()
            if rand < 0.6:
                return random.choice(self.cities_major)[0]
            elif rand < 0.9:
                return random.choice(self.cities_medium)[0]
            else:
                return random.choice(self.cities_international)[0]
        
        elif detected_type == 'state':
            if examples:
                return random.choice(examples)
            # Get state from city if available
            if 'city' in current_sample:
                city_name = current_sample['city']
                for city, state, _ in self.cities_major + self.cities_medium:
                    if city == city_name:
                        return state
            return self.fake.state_abbr()
        
        elif detected_type == 'country':
            if examples:
                return random.choice(examples)
            # 85% USA, 15% international
            if random.random() < 0.85:
                return "USA"
            else:
                return random.choice([city[2] for city in self.cities_international])
        
        elif detected_type == 'zip':
            return self.fake.zipcode()
        
        # === BUSINESS FIELDS ===
        elif detected_type == 'company':
            if examples:
                return random.choice(examples)
            # Choose by industry
            industry = random.choice(['tech', 'retail', 'finance'])
            if industry == 'tech':
                return random.choice(self.companies_tech)
            elif industry == 'retail':
                return random.choice(self.companies_retail)
            else:
                return random.choice(self.companies_finance)
        
        elif detected_type == 'department':
            if examples:
                return random.choice(examples)
            return random.choice(self.departments)
        
        elif detected_type == 'job':
            if examples:
                return random.choice(examples)
            # Get job based on department if available
            if 'department' in current_sample:
                dept = current_sample['department']
                if dept in self.job_titles:
                    return random.choice(self.job_titles[dept])
            return self.fake.job()
        
        elif detected_type == 'salary':
            # Realistic salary based on job title
            base = 50000
            if 'job' in current_sample or 'title' in current_sample:
                title = current_sample.get('job') or current_sample.get('title', '')
                title_lower = title.lower()
                if any(x in title_lower for x in ['senior', 'lead', 'principal', 'staff']):
                    base = 120000
                elif any(x in title_lower for x in ['manager', 'director']):
                    base = 150000
                elif 'vp' in title_lower or 'vice president' in title_lower:
                    base = 200000
                elif 'engineer' in title_lower or 'developer' in title_lower:
                    base = 90000
            
            # Add variance
            salary = base + random.randint(-15000, 35000)
            return round(salary / 1000) * 1000  # Round to nearest 1000
        
        # === PRODUCT FIELDS ===
        elif detected_type == 'product':
            if examples:
                return random.choice(examples)
            category = random.choice(['electronics', 'clothing'])
            if category == 'electronics':
                return random.choice(self.products_electronics)
            else:
                return random.choice(self.products_clothing)
        
        elif detected_type == 'category':
            if examples:
                return random.choice(examples)
            cat = random.choice(list(self.categories.keys()))
            # Sometimes return subcategory (40% of the time)
            if random.random() < 0.4:
                return f"{cat} > {random.choice(self.categories[cat])}"
            return cat
        
        elif detected_type == 'price':
            min_val = minimum if minimum is not None else 9.99
            max_val = maximum if maximum is not None else 999.99
            
            # Realistic pricing (tends to end in .99, .95, .49, .00)
            price = random.uniform(float(min_val), float(max_val))
            
            # Make it end realistically
            endings = [0.99, 0.95, 0.49, 0.00, 0.50]
            ending = random.choice(endings)
            price = int(price) + ending
            
            return round(price, 2)
        
        elif detected_type == 'quantity':
            min_val = minimum if minimum is not None else 0
            max_val = maximum if maximum is not None else 1000
            return random.randint(int(min_val), int(max_val))
        
        # === ID FIELDS ===
        elif detected_type == 'id':
            if 'int' in field_type.lower() or 'number' in field_type.lower():
                # Numeric ID
                return index + 1
            else:
                # String ID
                prefix = field_name.upper().replace('_', '')[:3]
                return f"{prefix}-{index+1:06d}"
        
        elif detected_type == 'uuid':
            return self.fake.uuid4()
        
        # === DATE FIELDS ===
        elif detected_type == 'created' or detected_type == 'date' or field_type in ['date', 'datetime', 'timestamp']:
            if 'birth' in field_lower or 'dob' in field_lower:
                # Birth date: 18-80 years ago
                start = datetime.now() - timedelta(days=365*80)
                end = datetime.now() - timedelta(days=365*18)
            elif 'created' in field_lower or 'registered' in field_lower or 'joined' in field_lower:
                # Account creation: past 1-5 years
                start = datetime.now() - timedelta(days=365*5)
                end = datetime.now()
            elif 'updated' in field_lower or 'modified' in field_lower:
                # Recent updates: past 1-30 days
                start = datetime.now() - timedelta(days=30)
                end = datetime.now()
            elif 'expiry' in field_lower or 'expire' in field_lower:
                # Future dates: 1-24 months from now
                start = datetime.now() + timedelta(days=30)
                end = datetime.now() + timedelta(days=730)
            else:
                # Generic: past year
                start = datetime.now() - timedelta(days=365)
                end = datetime.now()
            
            dt = self._random_datetime(start, end)
            return dt.isoformat()
        
        # === STATUS FIELDS ===
        elif detected_type == 'status':
            if examples:
                # If examples provided, use realistic distribution
                if len(examples) > 1:
                    weights = [0.7] + [0.3 / (len(examples) - 1)] * (len(examples) - 1)
                    return random.choices(examples, weights=weights)[0]
                return random.choice(examples)
            
            # Transaction status with realistic distribution
            if any(x in field_lower for x in ['transaction', 'payment', 'order']):
                return random.choices(
                    list(self.transaction_statuses.keys()),
                    weights=list(self.transaction_statuses.values())
                )[0]
            
            # Generic status
            return random.choice(["Active", "Inactive", "Pending", "Completed"])
        
        elif detected_type == 'active' or field_type == 'boolean':
            # Active/enabled fields are mostly True
            if any(x in field_lower for x in ['active', 'enabled', 'verified', 'approved']):
                return random.choices([True, False], weights=[0.85, 0.15])[0]
            return random.choice([True, False])
        
        # === NUMERIC FIELDS ===
        elif field_type in ['integer', 'int']:
            min_val = minimum if minimum is not None else 0
            max_val = maximum if maximum is not None else 100
            
            # Age fields
            if 'age' in field_lower:
                min_val = 18 if minimum is None else minimum
                max_val = 75 if maximum is None else maximum
            
            # Year fields
            elif 'year' in field_lower:
                min_val = 1950 if minimum is None else minimum
                max_val = datetime.now().year if maximum is None else maximum
            
            # Rating/Score fields (realistic distribution - bell curve)
            elif any(x in field_lower for x in ['rating', 'score']):
                min_val = 1 if minimum is None else minimum
                max_val = 5 if maximum is None else maximum
                # Bell curve around 3.5-4.5
                return max(min_val, min(max_val, int(random.gauss(4, 0.8))))
            
            return random.randint(int(min_val), int(max_val))
        
        elif field_type in ['float', 'number', 'decimal']:
            min_val = minimum if minimum is not None else 0.0
            max_val = maximum if maximum is not None else 100.0
            
            # Price fields
            if any(x in field_lower for x in ['price', 'amount', 'cost']):
                return self._generate_perfect_value(
                    field_name, 'price', examples, minimum, maximum,
                    description, unique, constraints, index, current_sample
                )
            
            # Percentage fields
            elif 'percent' in field_lower or 'rate' in field_lower:
                min_val = 0.0 if minimum is None else minimum
                max_val = 100.0 if maximum is None else maximum
                value = random.uniform(float(min_val), float(max_val))
                return round(value, 2)
            
            # Generic float
            value = random.uniform(float(min_val), float(max_val))
            return round(value, 2)
        
        # === STRING FIELDS ===
        elif field_type == 'string':
            # Use examples if provided
            if examples:
                return random.choice(examples)
            
            # Description/text fields
            if any(x in field_lower for x in ['description', 'comment', 'note', 'text', 'bio']):
                sentences = random.randint(2, 5)
                return self.fake.text(max_nb_chars=sentences * 50)
            
            # Skill fields
            elif 'skill' in field_lower:
                return random.choice(self.skills_technical + self.skills_soft)
            
            # URL fields
            elif 'url' in field_lower or 'website' in field_lower or 'link' in field_lower:
                return self.fake.url()
            
            # Generic string
            return self.fake.word().capitalize()
        
        # === FALLBACK ===
        else:
            if examples:
                return random.choice(examples)
            return None
    
    def _random_datetime(self, start: datetime, end: datetime) -> datetime:
        """Generate random datetime between start and end"""
        delta = end - start
        random_seconds = random.randint(0, int(delta.total_seconds()))
        return start + timedelta(seconds=random_seconds)
    
    def _format_output(self, samples: List[Dict[str, Any]], format: str) -> Any:
        """Format output data perfectly"""
        
        if format == "json":
            return samples
        
        elif format == "csv":
            if not samples:
                return ""
            
            import csv
            import io
            
            output = io.StringIO(newline='')  # 🔥 FIX: Prevent blank lines on Windows
            
            # Get all unique keys from all samples (handle missing fields)
            all_keys = []
            for sample in samples:
                for key in sample.keys():
                    if key not in all_keys:
                        all_keys.append(key)
            
            writer = csv.DictWriter(output, fieldnames=all_keys, extrasaction='ignore')
            writer.writeheader()
            
            # Write rows with proper handling of None values
            for sample in samples:
                row = {k: (v if v is not None else '') for k, v in sample.items()}
                writer.writerow(row)
            
            return output.getvalue()
        
        elif format == "sql":
            if not samples:
                return ""
            
            table_name = "synthetic_data"
            
            # Get all columns
            all_columns = []
            for sample in samples:
                for col in sample.keys():
                    if col not in all_columns:
                        all_columns.append(col)
            
            sql_statements = []
            
            for sample in samples:
                values = []
                for col in all_columns:
                    v = sample.get(col)
                    if v is None:
                        values.append("NULL")
                    elif isinstance(v, str):
                        # Escape single quotes properly
                        escaped = v.replace("'", "''")
                        values.append(f"'{escaped}'")
                    elif isinstance(v, bool):
                        values.append("TRUE" if v else "FALSE")
                    elif isinstance(v, (int, float)):
                        values.append(str(v))
                    else:
                        values.append(f"'{str(v)}'")
                
                sql = f"INSERT INTO {table_name} ({', '.join(all_columns)}) VALUES ({', '.join(values)});"
                sql_statements.append(sql)
            
            return "\n".join(sql_statements)
        
        else:
            return samples