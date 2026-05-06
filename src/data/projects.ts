export interface Project {
  title: string;
  year: number;
  description: string;
  links?: { label: string; href: string }[];
}

export const projects: Project[] = [
  {
    title: 'Example project',
    year: 2026,
    description:
      'Short description of the project — what it is, what it does, what makes it interesting. Replace this with your own.',
    links: [
      { label: 'code', href: 'https://github.com/' },
      { label: 'demo', href: 'https://example.com/' },
    ],
  },
];
