export interface Publication {
  title: string;
  authors: string;
  venue: string;
  year: number;
  links?: { label: string; href: string }[];
  note?: string;
}

export const publications: Publication[] = [
  {
    title: 'Example paper title goes here',
    authors: 'A. Author, B. Author, S. Guha',
    venue: 'Conference / Journal Name',
    year: 2025,
    links: [
      { label: 'pdf', href: 'https://arxiv.org/' },
      { label: 'arxiv', href: 'https://arxiv.org/' },
    ],
  },
];
