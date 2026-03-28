import re

newly_completed_tasks = [
    "2.3", "3.5", "3.6", "3.7", "3.8", "3.9",
    "4.6", "4.7"
]

with open("docs/TODO.md", "r", encoding="utf-8") as f:
    content = f.read()

for task in newly_completed_tasks:
    content = re.sub(rf"- \[ \] \*\*{task}\*\*", f"- [x] **{task}**", content)

with open("docs/TODO.md", "w", encoding="utf-8") as f:
    f.write(content)

print("TODO updated second time!")
