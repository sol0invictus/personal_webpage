export type PublicationCategory = 'nlp' | 'physics';

export interface Publication {
  title: string;
  authors: string;
  venue: string;
  year: number;
  category: PublicationCategory;
  links?: { label: string; href: string }[];
  note?: string;
}

export const categoryLabels: Record<PublicationCategory, string> = {
  nlp: 'NLP / Machine Learning',
  physics: 'Physics',
};

export const categoryOrder: PublicationCategory[] = ['nlp', 'physics'];

export const publications: Publication[] = [
  {
    title: 'Amazon Nova 2: Multimodal reasoning and generation models',
    authors: 'Amazon AGI Intelligence',
    venue: 'Technical Report, Amazon',
    year: 2025,
    category: 'nlp',
  },
  {
    title: 'CFT Correlators and Analytics',
    authors: 'S. Guha',
    venue: 'Ph.D. thesis, Texas A&M University',
    year: 2020,
    category: 'physics',
    note: 'doctoral dissertation',
  },
  {
    title: 'Resummation at finite conformal spin',
    authors: 'C. Cardona, S. Guha, S. K. Kanumilli, K. Sen',
    venue: 'Journal of High Energy Physics 2019(1)',
    year: 2019,
    category: 'physics',
  },
  {
    title: 'Relating diagrammatic expansion with conformal correlator expansion',
    authors: 'S. Guha, K. Sen',
    venue: 'arXiv preprint',
    year: 2019,
    category: 'physics',
    links: [{ label: 'arxiv', href: 'https://arxiv.org/abs/1911.11188' }],
  },
  {
    title: 'Correlators of mixed symmetry operators in defect CFTs',
    authors: 'S. Guha, B. Nagaraj',
    venue: 'Journal of High Energy Physics 2018(10)',
    year: 2018,
    category: 'physics',
  },
  {
    title: 'Eleven-dimensional supergravity in 4D, N=1 superspace',
    authors: 'K. Becker, M. Becker, D. Butter, S. Guha, W. D. Linch, D. Robbins',
    venue: 'Journal of High Energy Physics 2017(11)',
    year: 2017,
    category: 'physics',
  },
  {
    title: 'M-theory potential from the Hitchin functional in superspace',
    authors: 'K. Becker, M. Becker, S. Guha, W. D. Linch, D. Robbins',
    venue: 'Advances in Theoretical and Mathematical Physics 21(7)',
    year: 2017,
    category: 'physics',
  },
  {
    title: 'M-theory potential from the G₂ Hitchin functional in superspace',
    authors: 'K. Becker, M. Becker, S. Guha, W. D. Linch, D. Robbins',
    venue: 'Journal of High Energy Physics 2016(12)',
    year: 2016,
    category: 'physics',
  },
  {
    title: 'Physics at a 100 TeV pp collider: beyond the Standard Model phenomena',
    authors: 'T. Golling, M. Hance, P. Harris, M. L. Mangano, M. McCullough, F. Moortgat, et al.',
    venue: 'CERN Yellow Report',
    year: 2016,
    category: 'physics',
    links: [{ label: 'arxiv', href: 'https://arxiv.org/abs/1606.00947' }],
  },
  {
    title: 'Micro-Note on solving super p=2 form AMN',
    authors: 'S. Guha',
    venue: 'Note',
    year: 2016,
    category: 'physics',
  },
  {
    title: 'Collider interplay for supersymmetry, Higgs and dark matter',
    authors: 'O. Buchmueller, M. Citron, J. Ellis, S. Guha, J. Marrouche, K. A. Olive, et al.',
    venue: 'The European Physical Journal C 75(10)',
    year: 2015,
    category: 'physics',
  },
];
