# Reflection

## What I set out to do
As the start of my first term at SFU was approaching, I wanted a clear snapshot of the undergraduate faculties and programs.
Which programs were most popular? Which had the highest paid graduates? Which faculties had the biggest gender skews?
Wanting an answer, I searched around until I found the data SFU collects. 
The problem was that it was inconvenient to find and hard to read.
No one I knew would read hundreds of Excel rows just to find the least popular programs.
I wanted to create an easily accessible and interactive website where with a quick glance others could 
take in important information that could help them see the bigger picture of programs and outcomes.

## What went well
- I learned how to use the tools and modules Python has for data processing and visualizing such as
Pandas, Plotly, and PDFPlumber. 
- At first, I was cleaning Pandas `dfs` by iterating rows of the initial `dfs`. But by the end of this project,
I was able to learn how to properly clean `dfs` by using the appropriate methods like `drop`, `fill`, and `groupby`.
- I learned how to extract data from PDF files! Since all the outcome PDF files had the same format, I used my
own methods of using keywords to tell me where the data I wanted was.
- I practiced correct Python project structure, used a `venv`, and even added a small unit test.
- I learned how to use Flask to make and host a website using Python.

## What I would improve next time
- The outcomes PDF extraction was very inefficient taking on average 3-5 seconds. I saved my cleaned stats
in a JSON file to make my site load fast - needing only to clean my stats when changes were made.
Still, in the future I may have much larger PDF data sets and being able to clean them efficiently is something I look
forward to exploring.
- I only made 1 test for this project and I made it at the end. In the future, I want to start making
tests as I implement features and not at the very end. I also want to get better at making tests
because right now I feel overwhelmed with the amount of detail that a test could have.
- The graph on my site was implemented using plotly and while it is an amazing tool, I felt it lacked
customization and I would've liked more control.

## Conclusion
I am very happy with how this project turned out and how useful this site is. My favourite part was 
gaining a new perspective of SFU. Being in STEM, I rarely think about other faculties, so seeing how Education
has an unemployment rate of only 1.3% expanded my view of both SFU and the world.
I hope to share this project with other SFU students to help broaden our awareness of each other's fields.
