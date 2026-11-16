import './globals.css';
import type { Metadata } from 'next';
export const metadata: Metadata = {title:'Quantiva | Portfolio Intelligence',description:'Machine learning portfolio optimization and historical risk analysis'};
export default function RootLayout({children}:{children:React.ReactNode}){return <html lang="en"><body>{children}</body></html>}
