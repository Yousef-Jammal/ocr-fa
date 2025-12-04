// Dataset Templates Library

export const DATASET_TEMPLATES = {
  ecommerce: {
    name: "🛍️ E-Commerce Customers",
    icon: "🛒",
    description: "Customer purchase data with demographics",
    schema: {
      customer_id: { type: "string" },
      full_name: { type: "string" },
      email: { type: "email" },
      age: { type: "integer", minimum: 18, maximum: 75 },
      country: { type: "string", examples: ["USA", "UK", "Canada", "Australia"] },
      total_spent: { type: "float", minimum: 10.0, maximum: 50000.0 },
      signup_date: { type: "datetime" },
      is_premium: { type: "boolean" }
    },
    samples: 1000,
    prompts: [
      "modern e-commerce website product page",
      "online shopping cart interface",
      "retail store product display"
    ]
  },
  
  hr: {
    name: "👔 Employee Records",
    icon: "💼",
    description: "HR management and payroll data",
    schema: {
      employee_id: { type: "string" },
      full_name: { type: "string" },
      email: { type: "email" },
      department: { type: "string", examples: ["Engineering", "Sales", "Marketing", "HR"] },
      position: { type: "string" },
      salary: { type: "float", minimum: 30000, maximum: 250000 },
      hire_date: { type: "datetime" },
      performance_score: { type: "float", minimum: 1.0, maximum: 5.0 }
    },
    samples: 500,
    prompts: [
      "professional office workplace",
      "corporate meeting room",
      "business team collaboration"
    ]
  },
  
  healthcare: {
    name: "🏥 Patient Records",
    icon: "⚕️",
    description: "Medical patient data (synthetic)",
    schema: {
      patient_id: { type: "string" },
      full_name: { type: "string" },
      date_of_birth: { type: "datetime" },
      blood_type: { type: "string", examples: ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"] },
      height_cm: { type: "float", minimum: 150, maximum: 200 },
      weight_kg: { type: "float", minimum: 40, maximum: 150 },
      last_visit: { type: "datetime" }
    },
    samples: 300,
    prompts: [
      "modern hospital interior bright lighting",
      "medical clinic waiting room",
      "healthcare facility clean environment"
    ]
  },
  
  iot: {
    name: "📡 IoT Sensor Data",
    icon: "🌐",
    description: "Internet of Things device readings",
    schema: {
      sensor_id: { type: "string" },
      device_type: { type: "string", examples: ["Temperature", "Humidity", "Motion", "Light"] },
      location: { type: "string" },
      temperature: { type: "float", minimum: -10, maximum: 50 },
      battery_level: { type: "integer", minimum: 0, maximum: 100 },
      status: { type: "string", examples: ["Active", "Idle", "Warning", "Error"] },
      timestamp: { type: "datetime" }
    },
    samples: 2000,
    prompts: [
      "industrial warehouse with sensors",
      "smart building automation system",
      "IoT device control panel"
    ]
  },
  
  finance: {
    name: "💰 Financial Transactions",
    icon: "💳",
    description: "Banking and payment records",
    schema: {
      transaction_id: { type: "string" },
      account_holder: { type: "string" },
      amount: { type: "float", minimum: 0.01, maximum: 10000 },
      currency: { type: "string", examples: ["USD", "EUR", "GBP", "JPY"] },
      merchant: { type: "string" },
      category: { type: "string", examples: ["Shopping", "Food", "Travel", "Bills"] },
      timestamp: { type: "datetime" },
      is_fraudulent: { type: "boolean" }
    },
    samples: 5000,
    prompts: [
      "modern banking app interface",
      "payment terminal transaction",
      "financial dashboard analytics"
    ]
  },
  
  social: {
    name: "📱 Social Media Posts",
    icon: "💬",
    description: "Social network engagement data",
    schema: {
      post_id: { type: "string" },
      username: { type: "string" },
      content_type: { type: "string", examples: ["Text", "Image", "Video", "Story"] },
      likes: { type: "integer", minimum: 0, maximum: 100000 },
      comments: { type: "integer", minimum: 0, maximum: 10000 },
      shares: { type: "integer", minimum: 0, maximum: 5000 },
      engagement_rate: { type: "float", minimum: 0, maximum: 100 },
      posted_at: { type: "datetime" }
    },
    samples: 1000,
    prompts: [
      "social media app interface modern design",
      "content creator studio setup",
      "smartphone social networking"
    ]
  }
}

export type Template = typeof DATASET_TEMPLATES[keyof typeof DATASET_TEMPLATES]
