import '../styles/globals.css'
import NavBar from '../components/navbarstuff/navbar'
import Bottombar from '../components/Bottombar'
import React from 'react'
import { ChakraProvider } from '@chakra-ui/react'

function MyApp({ Component, pageProps }) {
    return (
        <>
        <div className="text-druk">
            <ChakraProvider>
                <NavBar />
            </ChakraProvider>
            </div>
            <Component {...pageProps} />
            <Bottombar />
        </>
    )
}

export default MyApp
