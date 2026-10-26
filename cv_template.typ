#set page(
  paper: "a4",
  margin: (x: 1.5cm, y: 1.5cm),
)

#set text(
  font: ("Segoe UI", "Arial", "Liberation Sans", "DejaVu Sans"),
  fallback: true,
  size: 10pt,
  fill: rgb("1a1a1a"),
  spacing: 120%,
)

#set par(
  justify: false,
  leading: 0.65em,
)

// Disable hyphenation to prevent "Full-Stack" from breaking into "Full-" and "Stack"
#set text(hyphenate: false)

#let section_heading(title) = {
  v(10pt)
  text(weight: "bold", size: 11pt, fill: rgb("0f172a"))[#upper(title)]
  v(-4pt)
  line(length: 100%, stroke: 0.5pt + rgb("cbd5e1"))
  v(4pt)
}

// --- HEADER ---
#align(center)[
  #text(size: 18pt, weight: "bold")[$name] \
  #v(2pt)
  #text(size: 11pt, weight: "medium", fill: rgb("475569"))[$title] \
  #v(2pt)
  #text(size: 9pt, fill: rgb("64748b"))[$location • $phone • #link("mailto:" + "$email_raw")[$email_display] • #link("$linkedin")[LinkedIn] • #link("$github")[GitHub]]
]

#v(8pt)

// --- PROFESSIONAL SUMMARY ---
#section_heading("Professional Summary")
[$summary]

// --- TECHNICAL SKILLS ---
#section_heading("Technical Skills")
[$skills_section]

// --- PROFESSIONAL EXPERIENCE ---
#section_heading("Professional Experience")
$experience_section

// --- EDUCATION ---
#section_heading("Education")
[$education_section]

// --- CERTIFICATIONS ---
#section_heading("Certifications")
[$certifications_section]