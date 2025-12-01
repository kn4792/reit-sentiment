# Capstone Presentation

This presentation covers the REIT Sentiment Analysis project using FinBERT.

## How to View

### Option 1: Open in Browser (Recommended)
1. Navigate to the file location:
   ```
   c:\Users\konid\Documents\Capstone\reit-sentiment-analysis\presentation.html
   ```
2. Double-click the file to open in your default browser
3. Use arrow keys or click to navigate:
   - **→** or **Space**: Next slide
   - **←**: Previous slide
   - **Esc**: Overview mode
   - **F**: Fullscreen

### Option 2: Live Server (Best for Development)
```bash
cd c:\Users\konid\Documents\Capstone\reit-sentiment-analysis
python -m http.server 8000
```
Then open: `http://localhost:8000/presentation.html`

## Presentation Structure

1. **Title Slide** - Project name, authors, advisors
2. **Project Overview** - Research question, goals, key metrics
3. **Domain Model** - 8 primary entities and relationships
4. **Architecture Overview** - 5-stage pipeline diagram
5. **Layer 1: Data Collection** - Web scraping infrastructure
6. **Layer 2: Text Preprocessing** - Cleaning pipeline
7. **Layer 3: FinBERT Sentiment** - Transformer-based classification
8. **Layer 4: Temporal Aggregation** - Firm-year transformation
9. **Layer 5: Econometric Modeling** - Panel regression specs
10. **Code Walkthrough** - GitHub repository structure (BREAK FOR LIVE CODE)
11. **Next Semester Plan** - Roadmap for completion
12. **Preliminary Results** - Descriptive findings and visualizations
13. **Methodological Contributions** - MNIR failure lessons
14. **Live Demo** - Terminal demonstration (BREAK FOR DEMO)
15. **Summary & Impact** - Achievements and contributions
16. **Questions** - Contact information

## Live Code Demonstration (Slide 10)

**Break from slides** to show actual code on GitHub:
- Show `scripts/clean_data.py` - TextCleaner class
- Show `scripts/finbert_sentiment.py` - GPU processing loop
- Show `scripts/aggregate_firm_year.py` - Aggregation logic
- Tie back to architecture diagram

Repository: https://github.com/kn4792/reit-sentiment-analysis

## Demo Script (Slide 14)

**Break from slides** to run live demonstration:

```bash
# Navigate to project
cd c:\Users\konid\Documents\Capstone\reit-sentiment-analysis

# Show FinBERT processing
python scripts/finbert_sentiment.py

# Show aggregation
python scripts/aggregate_firm_year.py

# Show descriptive analysis with visualizations
python scripts/descriptive_analysis.py
```

## Design Features

- **Modern Tech Stack**: reveal.js for presentations
- **Premium Design**: Gradient backgrounds, glassmorphism effects
- **Smooth Animations**: Fade-in transitions, hover effects
- **Professional Typography**: Inter font family for text, Fira Code for code
- **Responsive Layout**: Works on different screen sizes
- **Interactive Elements**: Clickable GitHub links, metric boxes
- **IEEE Conference Theme**: Professional academic styling

## Timing

- **Total duration**: 10 minutes
- **Slides**: ~30 seconds each average
- **Code demo**: ~2 minutes (Slide 10)
- **Live demo**: ~2 minutes (Slide 14)
- **Questions**: Built-in time

## Tips for Presentation

1. **Start strong** with the title slide showing all team members
2. **Emphasize metrics** (28,486 reviews, 91 REITs, 808 firm-years)
3. **Walk through architecture** layer by layer
4. **Code demo**: Focus on TextCleaner and FinBERT processing
5. **Preliminary results**: Highlight heterogeneous effects
6. **MNIR failure**: Turn negative result into learning
7. **End strong**: 80% complete, awaiting only WRDS data

## Key Messages

1. **Novel contribution**: First FinBERT application to REIT employee sentiment
2. **Production quality**: 95%+ test coverage, reproducible pipeline
3. **Preliminary evidence**: AI adoption has heterogeneous effects
4. **Methodological rigor**: Documented what doesn't work (MNIR)
5. **Nearly complete**: Only econometric analysis pending WRDS data

## Contact

- **Student**: Konain Niaz (kn4792@rit.edu)
- **Advisor**: Dr. Travis Desell (tjdvse@rit.edu)
- **Sponsor**: Dr. Debanjana Dey (ddey@saunders.rit.edu)
