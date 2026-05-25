export interface Project {
  title: string;
  year: number;
  description: string;
  links?: { label: string; href: string; sibling?: string }[];
}

export const projects: Project[] = [
  {
    title: 'Origin Map',
    year: 2026,
    description:
      'An interactive map of origin myths from cultures around the world. Browse creation stories, cosmogonies, and "how the world began" narratives by region, with each entry sourced and cross-referenced. A long-running side project of mine that grew out of reading too much comparative mythology.',
    links: [
      { label: 'site', href: 'https://originmap.eosaten.workers.dev/', sibling: 'originmap' },
    ],
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
