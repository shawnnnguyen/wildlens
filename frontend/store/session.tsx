import React, { createContext, useContext, useState } from 'react';

export type Scenario = 'identified' | 'safety' | 'unclear';

export interface SessionState {
  photoUri:    string;
  species:     string;
  taxon:       string;
  traits:      string[];
  confidence:  number;
  isDangerous: boolean;
  scenario:    Scenario | null;
  guideName:   string;
  setSession:  (s: Partial<Omit<SessionState, 'setSession'>>) => void;
}

const defaults: Omit<SessionState, 'setSession'> = {
  photoUri:    '',
  species:     '',
  taxon:       '',
  traits:      [],
  confidence:  0,
  isDangerous: false,
  scenario:    null,
  guideName:   'Baako',
};

const SessionContext = createContext<SessionState>({
  ...defaults,
  setSession: () => {},
});

export function SessionProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState(defaults);
  const setSession = (patch: Partial<typeof defaults>) =>
    setState(prev => ({ ...prev, ...patch }));
  return (
    <SessionContext.Provider value={{ ...state, setSession }}>
      {children}
    </SessionContext.Provider>
  );
}

export const useSession = () => useContext(SessionContext);

const MOCK_SUBJECTS = [
  { species: 'African Leopard',    taxon: 'Panthera pardus · Felidae',      traits: ['ROSETTE COAT','IN ACACIA TREE','ADULT'],   confidence: 0.92, isDangerous: false },
  { species: 'African Elephant',   taxon: 'Loxodonta africana · Elephantidae', traits: ['BULL','CLOSE PROXIMITY','EARS FLARED'], confidence: 0.88, isDangerous: true  },
  { species: 'Plains Zebra',       taxon: 'Equus quagga · Equidae',          traits: ['HERD','GRAZING','JUVENILE PRESENT'],      confidence: 0.79, isDangerous: false },
];

export function mockIdentify(uri: string): { scenario: Scenario; subject: typeof MOCK_SUBJECTS[0] } {
  const roll = Math.random();
  if (roll < 0.15) return { scenario: 'unclear',    subject: MOCK_SUBJECTS[0] };
  if (roll < 0.40) return { scenario: 'safety',     subject: MOCK_SUBJECTS[1] };
  const idx = Math.floor(Math.random() * MOCK_SUBJECTS.length);
  return { scenario: 'identified', subject: MOCK_SUBJECTS[idx] };
}
