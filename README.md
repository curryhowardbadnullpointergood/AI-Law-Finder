# AI Law Finder ⚖️🤖

**AI Law Finder** is a symbolic regression system built from scratch to discover **mathematical laws from raw data**.  
Given numerical datasets, the system uses **AI-inspired search + symbolic computation** to infer compact mathematical equations that describe the underlying relationships.  

This approach successfully rediscovered several classical physical laws — including **Newton’s Laws, Kepler’s Laws, and Simple Harmonic Motion** — directly from data.  This is probably the world's most configurable SR framework for this niche area, where you can quickly experiment with new startergies as proof of concept without having to dig through large open source projects, or old fortran legacy code from academic papers. 

---

## Project Goals
- Develop a system to **automatically discover physical or mathematical laws** from data.  
- Implement symbolic regression and genetic algorithms with minimal external dependencies.  
- Demonstrate extensibility: the method can be applied to **any domain with mathematically structured data**, not just physics.  

---

## Implementation
- **Languages:** Python (prototyping) · Go (efficient implementation)  
- **Dependencies:** NumPy, SymPy (for symbolic manipulation)  
- **Core Features:**
  - Genetic algorithm for evolving symbolic expressions  
  - Fitness evaluation based on data fit and complexity  
  - Symbolic simplification and expression tree manipulation  
  - Support for multiple strategies: random initialization, crossover, mutation  

---

## Results
The system was able to:  
- Rediscover **Newton’s Second Law**: `F = ma`  
- Rediscover **Kepler’s Laws of Planetary Motion**  
- Rediscover the equation of **Simple Harmonic Motion**  
- Achieve partial success with more complex systems (e.g. the pendulum)  

These results show the feasibility of **AI-driven scientific discovery**, even with a minimal implementation.  

---

## Learnings
- Gained deep understanding of symbolic regression, genetic algorithms, and computational mathematics.  
- Demonstrated the trade-off between **Python for rapid prototyping** and **Go for efficient execution**.  
- Explored the frontier of **interpretable AI** — models that output human-readable equations rather than black-box predictions.  

---

## Project Info
Type: Final-year research project (Independent)
Focus: Symbolic regression, AI, computational mathematics
Languages: Python, Go
Status: Research prototype, extensible to new domains (finance, biology, etc.)


