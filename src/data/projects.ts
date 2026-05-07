export interface Project {
  title: string;
  year: number;
  description: string;
  links?: { label: string; href: string }[];
}

export const projects: Project[] = [
  {
    title: 'Origin Map',
    year: 2026,
    description:
      'Coming soon — link will be added later.',
  },
  {
    title: 'nanoScaling',
    year: 2025,
    description:
      'A fork of nanoGPT built for scaling studies and optimizer research. Compares optimizers (Muon, Scion, AdamW) via activation statistics and weight-geometry metrics; supports SP / muP / CompleteP parametrizations, spectral logging, and DDP multi-GPU training on OpenWebText and FineWeb-Edu.',
    links: [
      { label: 'code', href: 'https://github.com/sol0invictus/nanoScaling' },
    ],
  },
];
